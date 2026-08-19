"""Training-data pipeline for IlluminaDepth-448.

What this module does
---------------------
It turns a plain directory of photographs into ``(rgb_448, teacher_disparity)``
pairs suitable for knowledge distillation, and it owns the exact preprocessing
the deployed WebGPU pipeline performs so the student never sees a different
input distribution at training time than at inference time.

Why distillation
----------------
IlluminaDepth-448 is a 1.39 M parameter, 2.2 GFLOP network. There is no chance
of training something that small from scratch on the small, noisy, mutually
inconsistent public depth datasets and getting a usable relative-depth field.
The practical route is the one MiDaS and Depth Anything themselves use for
their small variants: freeze a large teacher, run it over a large pile of
unlabelled ordinary images, and regress its (scale- and shift-free) disparity
with a scale-and-shift-invariant loss. The teacher's output is not metric
truth; it is a self-consistent relative-depth field, which is exactly what the
lighting pass downstream needs.

The pieces
----------
``find_images``            recursive image discovery.
``srgb_to_linear``         the exact piecewise sRGB EOTF (not gamma 2.2).
``make_network_input``     sRGB [0,1] -> the 4-channel tensor the stem expects.
``AugmentConfig`` /
``GeometricParams``        augmentation description and the sampled parameters,
                           so the *same* geometry can be replayed on the image
                           and on the target.
``color_jitter``           photometric augmentation, student input only.
``DepthTeacher``           interface; ``DepthAnythingV2Teacher`` is the real
                           one, ``SyntheticTeacher`` is a clearly marked
                           smoke-test stand-in.
``TeacherCache``           on-disk cache of teacher disparity, so the teacher
                           (which costs ~40x more FLOPs than a student training
                           step) runs once per image rather than once per epoch.
``DistillationDataset``    the ``torch.utils.data.Dataset``.

Preprocessing contract (must match ``src/model/reference.ts::refPreprocess``)
----------------------------------------------------------------------------
For each pixel, with ``v`` the sRGB-encoded component in [0, 1]::

    lin_i  = srgb_to_linear(v_i) * exposure
    out_i  = (lin_i - mean_i) / std_i          for i in {R, G, B}
    out_3  = 0.2126*lin_R + 0.7152*lin_G + 0.0722*lin_B

Note that the ImageNet mean/std are applied in **linear light**, not in sRGB.
That is unusual, and it is deliberate: the renderer this model feeds works in
linear light, so the network's input is the same tensor the lighting pass
already has in registers. Channel 3 carries luminance, which the joint
bilateral upsample reuses as its guide.

Geometry and the loss
---------------------
Geometric augmentation (random resized crop + horizontal flip) is applied to
the image *and* the target with identical parameters. Cropping a teacher
disparity map is sound even though relative depth is only defined up to an
affine transform per image: the training loss (MiDaS scale-and-shift-invariant
loss) quotients out exactly that ambiguity, so a crop of a global disparity map
is as good a target as a fresh teacher run on the crop. That is what makes the
disk cache legitimate rather than a corner cut.

Photometric augmentation (colour jitter, exposure jitter) is applied only to
the student's copy, after the teacher has seen the clean pixels.

Dependencies: torch, Pillow. ``transformers`` is imported lazily and only when
a real teacher is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

try:
    from PIL import Image, ImageFile, ImageOps
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "Pillow is required by illumina.data. Install it with:\n"
        "    pip install -r tools/requirements.txt"
    ) from exc

# Some scraped datasets contain JPEGs with a truncated final scan. Decoding
# what is there beats dropping the sample.
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = [
    "IMAGE_EXTENSIONS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LUMINANCE_WEIGHTS",
    "DEFAULT_TEACHER_ID",
    "AugmentConfig",
    "GeometricParams",
    "DepthTeacher",
    "DepthAnythingV2Teacher",
    "SyntheticTeacher",
    "TeacherUnavailableError",
    "TeacherCache",
    "DistillationDataset",
    "build_teacher",
    "color_jitter",
    "find_images",
    "make_network_input",
    "normalize_disparity",
    "precompute_teacher_cache",
    "sample_geometry",
    "srgb_to_linear",
]

IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)

# ImageNet statistics. Applied in linear light -- see the module docstring.
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

# Rec. 709 luminance, matching ``luminanceOf`` in reference.ts.
LUMINANCE_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)

DEFAULT_TEACHER_ID: Final[str] = "depth-anything/Depth-Anything-V2-Small-hf"

# Cache files are versioned so a change in how targets are produced invalidates
# every stale entry instead of silently poisoning a run.
CACHE_VERSION: Final[int] = 1


# ---------------------------------------------------------------------------
# Image discovery and decoding
# ---------------------------------------------------------------------------


def find_images(root: str | Path, recursive: bool = True) -> list[Path]:
    """Returns every image file under ``root``, sorted for reproducibility.

    Sorting matters: the train/val split is a deterministic function of the
    index, so an unsorted (filesystem-order) listing would silently reshuffle
    the split between machines.
    """
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise FileNotFoundError(f"image root {root_path} is not a directory")
    pattern = "**/*" if recursive else "*"
    paths = [
        p
        for p in root_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    if not paths:
        raise FileNotFoundError(
            f"no images with extensions {IMAGE_EXTENSIONS} found under {root_path}"
        )
    return paths


def load_image(path: str | Path) -> Tensor:
    """Decodes an image to a ``(3, H, W)`` float32 tensor of sRGB in [0, 1].

    EXIF orientation is applied, because roughly 15 % of phone photographs are
    stored rotated and a depth model trained on sideways images is a depth
    model trained on the wrong prior.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        width, height = img.size
        # bytearray() gives torch a writable buffer, which avoids both a
        # UserWarning and a dependency on numpy for the conversion.
        buf = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    hwc = buf.view(height, width, 3)
    return hwc.permute(2, 0, 1).float().div_(255.0)


# ---------------------------------------------------------------------------
# Preprocessing -- the exact contract of the deployed kernel
# ---------------------------------------------------------------------------


def srgb_to_linear(x: Tensor) -> Tensor:
    """Exact piecewise sRGB electro-optical transfer function.

    ``c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4``

    This is not the same as ``x ** 2.2``: the two differ by up to 0.02 in the
    shadows, which is where a depth network gets most of its shading cues.
    Both branches are always evaluated (``torch.where`` is not lazy), so the
    power branch clamps its input to keep the gradient finite for x < 0.
    """
    clamped = x.clamp_min(0.0)
    high = ((clamped + 0.055) / 1.055).pow(2.4)
    return torch.where(x <= 0.04045, x / 12.92, high)


def make_network_input(
    rgb_srgb: Tensor,
    exposure: float | Tensor = 1.0,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    decode_srgb: bool = True,
) -> Tensor:
    """sRGB image -> the 4-channel tensor the network's stem consumes.

    This is the batched, per-sample-exposure form of
    :func:`illumina.model.preprocess`, which is the canonical port of
    ``refPreprocess`` and the one ``tools/export/verify_parity.py`` checks
    against the TypeScript reference. The two compute the same function; this
    one additionally accepts a vector of exposures so a batch can carry a
    different exposure per image, which is what the exposure augmentation
    needs and what a scalar-only signature cannot express.

    Args:
        rgb_srgb: ``(3, H, W)`` or ``(N, 3, H, W)``, sRGB-encoded, in [0, 1].
        exposure: scalar, or a ``(N,)`` tensor of per-sample multipliers
            applied in linear light (the deployed kernel takes the same knob
            as a uniform).
        mean, std: normalisation statistics, applied in linear light.
        decode_srgb: set False if the source is already linear-light.

    Returns:
        ``(4, H, W)`` or ``(N, 4, H, W)``. Channels 0-2 are the normalised
        linear RGB; channel 3 is the unnormalised linear luminance, which the
        bilateral upsample reuses as its edge guide.
    """
    batched = rgb_srgb.dim() == 4
    x = rgb_srgb if batched else rgb_srgb.unsqueeze(0)
    if x.shape[1] != 3:
        raise ValueError(f"expected 3 colour channels, got shape {tuple(x.shape)}")

    lin = srgb_to_linear(x) if decode_srgb else x

    if isinstance(exposure, Tensor):
        exp = exposure.to(device=lin.device, dtype=lin.dtype).reshape(-1, 1, 1, 1)
    else:
        exp = torch.as_tensor(exposure, device=lin.device, dtype=lin.dtype)
    lin = lin * exp

    mean_t = torch.tensor(mean, device=lin.device, dtype=lin.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=lin.device, dtype=lin.dtype).view(1, 3, 1, 1)
    normalised = (lin - mean_t) / std_t

    luma_w = torch.tensor(
        LUMINANCE_WEIGHTS, device=lin.device, dtype=lin.dtype
    ).view(1, 3, 1, 1)
    luma = (lin * luma_w).sum(dim=1, keepdim=True)

    out = torch.cat([normalised, luma], dim=1)
    return out if batched else out.squeeze(0)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


@dataclass
class AugmentConfig:
    """Augmentation strength.

    Defaults are deliberately strong. Distillation targets are cheap and
    plentiful but they all come from one teacher, so the student's main failure
    mode is copying the teacher's texture cues rather than learning geometry.
    Aggressive cropping and photometric jitter is the standard antidote.
    """

    # Random resized crop: fraction of the source area to keep, and the
    # aspect-ratio range of the crop box.
    scale: tuple[float, float] = (0.25, 1.0)
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    hflip_prob: float = 0.5

    # Photometric, student input only. Each is a maximum symmetric deviation.
    brightness: float = 0.35
    contrast: float = 0.35
    saturation: float = 0.35
    hue: float = 0.05  # in turns, i.e. 0.05 == 18 degrees

    # Exposure multiplier applied in linear light, sampled as 2 ** U(-e, +e).
    exposure_log2: float = 0.6

    # Probability of applying the photometric block at all.
    jitter_prob: float = 0.8

    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        if not 0.0 < self.scale[0] <= self.scale[1] <= 1.0:
            raise ValueError(f"invalid scale range {self.scale}")
        if not 0.0 < self.ratio[0] <= self.ratio[1]:
            raise ValueError(f"invalid ratio range {self.ratio}")

    @classmethod
    def deterministic(cls) -> "AugmentConfig":
        """The validation transform: centred square crop, no jitter.

        With ``scale == (1, 1)`` and ``ratio == (1, 1)`` the sampler's fallback
        path takes the largest centred square, which is then resized to the
        network's input size. That is deterministic, it wastes no pixels on
        letterboxing, and it matches the training distribution -- training
        crops are constrained to an aspect near 1 and squashed to a square too.
        """
        return cls(
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            hflip_prob=0.0,
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
            hue=0.0,
            exposure_log2=0.0,
            jitter_prob=0.0,
        )


@dataclass(frozen=True)
class GeometricParams:
    """A sampled crop box, in source-image pixels, plus a flip flag.

    Kept as data rather than applied immediately so the identical geometry can
    be replayed on the image, on a cached teacher map at a different
    resolution, and (for debugging) on anything else derived from the source.
    """

    top: int
    left: int
    height: int
    width: int
    hflip: bool


def sample_geometry(
    height: int,
    width: int,
    cfg: AugmentConfig,
    rng: random.Random,
) -> GeometricParams:
    """Samples a random resized crop box, following torchvision's algorithm.

    Ten rejection-sampling attempts at the requested area/aspect, then a
    deterministic centred fallback that respects the aspect bounds. Reproduced
    here rather than imported so the pipeline needs no torchvision.
    """
    area = float(height * width)
    log_ratio = (math.log(cfg.ratio[0]), math.log(cfg.ratio[1]))

    for _ in range(10):
        target_area = area * rng.uniform(cfg.scale[0], cfg.scale[1])
        aspect = math.exp(rng.uniform(log_ratio[0], log_ratio[1]))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = rng.randint(0, height - crop_h)
            left = rng.randint(0, width - crop_w)
            return GeometricParams(
                top, left, crop_h, crop_w, rng.random() < cfg.hflip_prob
            )

    # Fallback: the largest centred box inside the aspect bounds.
    in_ratio = width / height
    if in_ratio < cfg.ratio[0]:
        crop_w = width
        crop_h = int(round(crop_w / cfg.ratio[0]))
    elif in_ratio > cfg.ratio[1]:
        crop_h = height
        crop_w = int(round(crop_h * cfg.ratio[1]))
    else:
        crop_w, crop_h = width, height
    crop_h = min(crop_h, height)
    crop_w = min(crop_w, width)
    return GeometricParams(
        (height - crop_h) // 2,
        (width - crop_w) // 2,
        crop_h,
        crop_w,
        rng.random() < cfg.hflip_prob,
    )


def apply_geometry(
    image: Tensor,
    geom: GeometricParams,
    size: int,
    mode: str = "bilinear",
) -> Tensor:
    """Crops, resizes to ``size x size`` and optionally flips a ``(C, H, W)``.

    ``geom`` is expressed in the coordinates of the tensor it was sampled from;
    when it is replayed on a tensor of a different resolution (a cached teacher
    map, say) the box is rescaled proportionally. That keeps the image and its
    target in registration to within half a cache pixel.
    """
    _, height, width = image.shape
    top = max(0, min(geom.top, height - 1))
    left = max(0, min(geom.left, width - 1))
    bottom = max(top + 1, min(geom.top + geom.height, height))
    right = max(left + 1, min(geom.left + geom.width, width))

    crop = image[:, top:bottom, left:right]
    resized = F.interpolate(
        crop.unsqueeze(0).float(),
        size=(size, size),
        mode=mode,
        align_corners=False,
        antialias=mode in ("bilinear", "bicubic"),
    ).squeeze(0)
    if geom.hflip:
        resized = torch.flip(resized, dims=(2,))
    return resized


def rescale_geometry(
    geom: GeometricParams,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> GeometricParams:
    """Maps a crop box from one resolution to another, preserving the flip."""
    sy = dst_h / src_h
    sx = dst_w / src_w
    top = int(round(geom.top * sy))
    left = int(round(geom.left * sx))
    height = max(1, int(round(geom.height * sy)))
    width = max(1, int(round(geom.width * sx)))
    return GeometricParams(top, left, height, width, geom.hflip)


# RGB <-> YIQ, used for hue rotation. Hue is rotated in the IQ plane, which is
# a linear operation and therefore differentiable, branch-free and fast --
# unlike torchvision's HSV round trip. The two agree to within about one
# degree of hue for the small rotations used here.
_RGB_TO_YIQ: Final[tuple[tuple[float, float, float], ...]] = (
    (0.299, 0.587, 0.114),
    (0.5959, -0.2746, -0.3213),
    (0.2115, -0.5227, 0.3112),
)
_YIQ_TO_RGB: Final[tuple[tuple[float, float, float], ...]] = (
    (1.0, 0.956, 0.619),
    (1.0, -0.272, -0.647),
    (1.0, -1.106, 1.703),
)
# Rec. 601 luminance, matching torchvision's grayscale conversion. This is
# *not* LUMINANCE_WEIGHTS: that one is Rec. 709 in linear light and belongs to
# the network's input, this one is a photometric-augmentation detail in sRGB.
_JITTER_GRAY: Final[tuple[float, float, float]] = (0.299, 0.587, 0.114)


def _hue_matrix(turns: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    angle = 2.0 * math.pi * turns
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = [
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a],
    ]
    to_yiq = torch.tensor(_RGB_TO_YIQ, device=device, dtype=dtype)
    to_rgb = torch.tensor(_YIQ_TO_RGB, device=device, dtype=dtype)
    rot_t = torch.tensor(rot, device=device, dtype=dtype)
    return to_rgb @ rot_t @ to_yiq


def color_jitter(image: Tensor, cfg: AugmentConfig, rng: random.Random) -> Tensor:
    """Brightness / contrast / saturation / hue jitter on a ``(3, H, W)`` sRGB.

    Applied in the order torchvision uses when it does not shuffle: brightness,
    contrast, saturation, hue, with a clamp to [0, 1] after each step so the
    result stays a valid image. Never applied to a depth target.
    """
    if cfg.jitter_prob <= 0.0 or rng.random() >= cfg.jitter_prob:
        return image

    out = image
    if cfg.brightness > 0.0:
        factor = 1.0 + rng.uniform(-cfg.brightness, cfg.brightness)
        out = (out * factor).clamp_(0.0, 1.0)

    gray_w = torch.tensor(_JITTER_GRAY, device=out.device, dtype=out.dtype)
    gray_w = gray_w.view(3, 1, 1)

    if cfg.contrast > 0.0:
        factor = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        mean = (out * gray_w).sum(dim=0, keepdim=True).mean()
        out = ((out - mean) * factor + mean).clamp_(0.0, 1.0)

    if cfg.saturation > 0.0:
        factor = 1.0 + rng.uniform(-cfg.saturation, cfg.saturation)
        gray = (out * gray_w).sum(dim=0, keepdim=True)
        out = ((out - gray) * factor + gray).clamp_(0.0, 1.0)

    if cfg.hue > 0.0:
        turns = rng.uniform(-cfg.hue, cfg.hue)
        matrix = _hue_matrix(turns, out.device, out.dtype)
        flat = out.reshape(3, -1)
        out = (matrix @ flat).reshape(out.shape).clamp_(0.0, 1.0)

    return out


# ---------------------------------------------------------------------------
# Teachers
# ---------------------------------------------------------------------------


class TeacherUnavailableError(RuntimeError):
    """Raised when the requested teacher cannot be constructed.

    Deliberately loud and specific: a silent fallback to a fake teacher would
    produce a training run that looks healthy for hours and yields a model that
    has learned nothing about depth.
    """


class DepthTeacher:
    """Interface for a frozen teacher that maps images to disparity.

    Implementations take ``(N, 3, H, W)`` sRGB in [0, 1] and return
    ``(N, 1, H, W)`` relative inverse depth (disparity), larger meaning nearer,
    already normalised to roughly [0, 1] per image. The absolute scale is
    meaningless -- the training loss is invariant to an affine transform of the
    target -- but keeping it in [0, 1] means the student's sigmoid output does
    not have to fight its own saturation early in training.
    """

    #: Human-readable identifier, recorded in checkpoints and the cache key.
    name: str = "abstract"

    def __call__(self, images_srgb: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError


def normalize_disparity(
    disparity: Tensor,
    robust: bool = True,
    lo_q: float = 0.02,
    hi_q: float = 0.98,
    chunk: int = 8,
) -> Tensor:
    """Per-image affine normalisation of disparity into [0, 1].

    With ``robust`` the 2nd and 98th percentiles are used instead of the exact
    min and max, so a single blown-out sky pixel or a speck of sensor noise
    cannot compress the entire useful range into a sliver. Values outside the
    percentile range are kept (not clipped) because they are real geometry; the
    loss sees them as slightly out-of-range targets, which is harmless.

    Batches are quantiled in chunks because ``torch.quantile`` refuses inputs
    above roughly 16 M elements.
    """
    flat = disparity.reshape(disparity.shape[0], -1).float()
    if robust:
        lows: list[Tensor] = []
        highs: list[Tensor] = []
        q = torch.tensor([lo_q, hi_q], device=flat.device, dtype=flat.dtype)
        for start in range(0, flat.shape[0], chunk):
            part = flat[start : start + chunk]
            bounds = torch.quantile(part, q, dim=1)
            lows.append(bounds[0])
            highs.append(bounds[1])
        low = torch.cat(lows)
        high = torch.cat(highs)
    else:
        low = flat.amin(dim=1)
        high = flat.amax(dim=1)

    span = (high - low).clamp_min(1e-6)
    shape = (-1,) + (1,) * (disparity.dim() - 1)
    return (disparity - low.reshape(shape)) / span.reshape(shape)


class DepthAnythingV2Teacher(DepthTeacher):
    """Depth Anything V2 Small, loaded through HuggingFace transformers.

    The model is a ViT-S/14 DINOv2 encoder with a DPT head. It emits relative
    inverse depth: larger is nearer, and the field is only defined up to scale
    and shift, which is precisely the invariance our loss assumes.

    We read the normalisation constants and the working resolution from the
    published image processor, then do the resize and normalise ourselves on
    tensors already on the GPU. Going through the processor's PIL path would
    force a device round trip and single-image batching, which costs far more
    than it is worth given the teacher dominates the FLOP budget.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_TEACHER_ID,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        resolution: int | None = None,
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise TeacherUnavailableError(
                "The 'transformers' package is not installed, so the Depth "
                "Anything V2 teacher cannot be loaded.\n"
                "Install it with:\n"
                "    pip install 'transformers>=4.45' 'huggingface_hub>=0.25' "
                "safetensors\n"
                "Then re-run, or pass --teacher synthetic for a pipeline "
                "smoke test that does NOT train a usable model."
            ) from exc

        self.name = model_id
        self.device = torch.device(device)
        self.dtype = dtype

        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id)
        except Exception as exc:  # network, auth, revision, disk -- all fatal
            raise TeacherUnavailableError(
                f"Could not load the teacher '{model_id}'.\n"
                f"Underlying error: {type(exc).__name__}: {exc}\n"
                "Common causes: no network access on this machine, or the "
                "model has not been downloaded yet. To pre-download it:\n"
                f"    huggingface-cli download {model_id}\n"
                "and set HF_HOME to a directory the training job can read."
            ) from exc

        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        size = getattr(processor, "size", None) or {}
        default_res = int(size.get("height", 518)) if isinstance(size, dict) else 518
        self.resolution = int(resolution or default_res)
        # DINOv2 patches are 14x14; a non-multiple resolution silently changes
        # the token grid and degrades the teacher.
        if self.resolution % 14 != 0:
            adjusted = max(14, int(round(self.resolution / 14)) * 14)
            warnings.warn(
                f"teacher resolution {self.resolution} is not a multiple of 14; "
                f"using {adjusted} instead",
                stacklevel=2,
            )
            self.resolution = adjusted

        mean = tuple(getattr(processor, "image_mean", IMAGENET_MEAN))
        std = tuple(getattr(processor, "image_std", IMAGENET_STD))
        self._mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, images_srgb: Tensor) -> Tensor:
        """``(N, 3, H, W)`` sRGB in [0, 1] -> ``(N, 1, H, W)`` disparity."""
        if images_srgb.dim() != 4 or images_srgb.shape[1] != 3:
            raise ValueError(
                f"teacher expects (N, 3, H, W), got {tuple(images_srgb.shape)}"
            )
        out_h, out_w = images_srgb.shape[-2:]
        x = images_srgb.to(device=self.device, dtype=torch.float32)
        x = F.interpolate(
            x,
            size=(self.resolution, self.resolution),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
        mean = self._mean.to(x.device)
        std = self._std.to(x.device)
        x = (x - mean) / std

        depth = self.model(pixel_values=x.to(self.dtype)).predicted_depth
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = depth.float()
        depth = F.interpolate(
            depth, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        return normalize_disparity(depth)


class SyntheticTeacher(DepthTeacher):
    """NOT A REAL TEACHER. Smoke-test stand-in only.

    Produces a smooth, deterministic pseudo-disparity from the image itself: a
    vertical gradient (the "floor is near, sky is far" prior) blended with a
    blurred luminance term. It exists so that ``train.py``, the checkpointing
    path and the export path can be exercised end to end on a machine with no
    network access and no transformers install.

    A model trained against this learns a vignette. Every code path that
    constructs one prints a warning, and the name is recorded in the
    checkpoint so a synthetic run cannot be mistaken for a real one later.
    """

    name = "synthetic"

    def __init__(self) -> None:
        warnings.warn(
            "SyntheticTeacher produces fake targets. Use it only to smoke-test "
            "the pipeline; the resulting weights are worthless for depth.",
            stacklevel=2,
        )

    @torch.inference_mode()
    def __call__(self, images_srgb: Tensor) -> Tensor:
        n, _, height, width = images_srgb.shape
        rows = torch.linspace(
            0.0, 1.0, height, device=images_srgb.device, dtype=torch.float32
        )
        vertical = rows.view(1, 1, height, 1).expand(n, 1, height, width)
        gray = images_srgb.mean(dim=1, keepdim=True).float()
        blurred = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        return normalize_disparity(0.75 * vertical + 0.25 * blurred)


def build_teacher(
    kind: str = "depth-anything-v2-small",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    model_id: str | None = None,
    resolution: int | None = None,
) -> DepthTeacher:
    """Constructs a teacher by name.

    ``kind`` is one of ``depth-anything-v2-small`` (the default and the only
    one that trains a real model), ``hf:<model id>`` for any other HuggingFace
    depth-estimation checkpoint, or ``synthetic`` for the smoke-test teacher.
    """
    if kind == "synthetic":
        return SyntheticTeacher()
    if kind.startswith("hf:"):
        return DepthAnythingV2Teacher(kind[3:], device, dtype, resolution)
    if kind in ("depth-anything-v2-small", "dav2-small", "default"):
        return DepthAnythingV2Teacher(
            model_id or DEFAULT_TEACHER_ID, device, dtype, resolution
        )
    raise ValueError(
        f"unknown teacher '{kind}'. Expected 'depth-anything-v2-small', "
        "'hf:<model-id>' or 'synthetic'."
    )


# ---------------------------------------------------------------------------
# Teacher cache
# ---------------------------------------------------------------------------


class TeacherCache:
    """On-disk cache of teacher disparity, one float16 tensor per source image.

    Why this exists: a Depth Anything V2 Small forward pass at 518x518 costs
    roughly 90 GFLOP, while a full training step of the student costs about
    6.6 GFLOP. Running the teacher inside the training loop therefore makes the
    *teacher* 90 % of the job and caps throughput at the teacher's speed for
    every epoch. Precomputing once turns a multi-epoch run from teacher-bound
    into student-bound.

    Layout: ``<cache_dir>/<sha1[:2]>/<sha1>.pt``, holding a dict with the
    disparity (float16, ``(1, h, w)``, long side capped at ``max_side``), the
    source resolution it was computed from, the teacher name and the cache
    version. The key hashes the absolute path, the file size and the mtime, so
    editing or replacing an image invalidates its entry.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        teacher_name: str,
        max_side: int = 512,
    ) -> None:
        self.dir = Path(cache_dir).expanduser()
        self.teacher_name = teacher_name
        self.max_side = int(max_side)
        self.dir.mkdir(parents=True, exist_ok=True)

    def key(self, path: Path) -> str:
        stat = path.stat()
        digest = hashlib.sha1()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(f"|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8"))
        digest.update(
            f"|{self.teacher_name}|{self.max_side}|{CACHE_VERSION}".encode("utf-8")
        )
        return digest.hexdigest()

    def path_for(self, path: Path) -> Path:
        key = self.key(path)
        return self.dir / key[:2] / f"{key}.pt"

    def has(self, path: Path) -> bool:
        return self.path_for(path).is_file()

    def store(self, path: Path, disparity: Tensor, src_hw: tuple[int, int]) -> None:
        """Writes one entry. ``disparity`` is ``(1, h, w)`` in [0, 1]."""
        target = self.path_for(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "teacher": self.teacher_name,
            "src_hw": tuple(int(v) for v in src_hw),
            "disparity": disparity.detach().to(torch.float16).cpu(),
        }
        # Write-then-rename so a killed job never leaves a half-written entry
        # that a later run would happily load as garbage.
        tmp = target.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(target)

    def load(self, path: Path) -> tuple[Tensor, tuple[int, int]]:
        """Returns ``(disparity (1, h, w) float32, source (H, W))``."""
        entry = torch.load(self.path_for(path), map_location="cpu")
        if entry.get("version") != CACHE_VERSION:
            raise ValueError(f"stale cache entry for {path}")
        return entry["disparity"].float(), tuple(entry["src_hw"])

    def cache_side(self, height: int, width: int) -> tuple[int, int]:
        """Resolution an entry is stored at: aspect preserved, long side capped."""
        longest = max(height, width)
        if longest <= self.max_side:
            return height, width
        scale = self.max_side / longest
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def precompute_teacher_cache(
    paths: Sequence[Path],
    teacher: DepthTeacher,
    cache: TeacherCache,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    log_every: int = 50,
) -> int:
    """Fills ``cache`` for every path that does not already have an entry.

    Returns the number of newly computed entries. Safe to interrupt and re-run:
    entries are written atomically and existing ones are skipped, so a job that
    dies at 60 % resumes at 60 %.
    """
    todo = [p for p in paths if not cache.has(p)]
    if not todo:
        print(f"teacher cache: all {len(paths)} entries already present")
        return 0

    print(
        f"teacher cache: {len(todo)} of {len(paths)} images to compute "
        f"into {cache.dir}"
    )
    done = 0
    # Entries are stored at the source aspect ratio, so only images that
    # happen to share a cached resolution can go through the teacher as one
    # batch. Grouping by shape within a window recovers most of the batching
    # win on the usual case (a dataset of uniformly sized photographs) without
    # distorting anything on a mixed one.
    window = max(batch_size * 8, batch_size)
    for start in range(0, len(todo), window):
        groups: dict[tuple[int, int], list[tuple[Path, Tensor, tuple[int, int]]]] = {}
        for path in todo[start : start + window]:
            try:
                img = load_image(path)
            except Exception as exc:
                warnings.warn(f"skipping unreadable image {path}: {exc}", stacklevel=2)
                continue
            src_h, src_w = img.shape[-2:]
            out_h, out_w = cache.cache_side(src_h, src_w)
            resized = F.interpolate(
                img.unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            groups.setdefault((out_h, out_w), []).append(
                (path, resized, (src_h, src_w))
            )

        for items in groups.values():
            for offset in range(0, len(items), batch_size):
                batch_items = items[offset : offset + batch_size]
                batch = torch.stack([item[1] for item in batch_items]).to(device)
                disparity = teacher(batch).cpu()
                for i, (path, _, src_hw) in enumerate(batch_items):
                    cache.store(path, disparity[i], src_hw)
                    done += 1
                if log_every and done % log_every < batch_size:
                    print(f"  {done}/{len(todo)}")
    print(f"teacher cache: wrote {done} entries")
    return done


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Counters a training run can print without reaching into the dataset."""

    decoded: int = 0
    failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class DistillationDataset(Dataset):
    """Produces ``(rgb_448, teacher_disparity)`` pairs from a directory.

    Each item is a dict with:

    ``student_rgb``  ``(3, S, S)`` float32 sRGB in [0, 1], photometrically
                     jittered. Feed this through :func:`make_network_input`
                     together with ``exposure``.
    ``teacher_rgb``  ``(3, S, S)`` float32 sRGB in [0, 1], the same crop
                     *without* photometric jitter. Only meaningful when the
                     teacher runs online; it is what the teacher must see.
    ``target``       ``(1, S, S)`` disparity in [0, 1] when a cache is in use,
                     otherwise a ``(0, S, S)`` empty tensor. The shape is the
                     same for every item in a run, so the default collate
                     stacks it without a custom function; the trainer tests
                     ``target.shape[1] == 1`` to decide whether it must run the
                     teacher itself.
    ``exposure``     scalar float32, the linear-light exposure multiplier that
                     was sampled for this item.
    ``index``        int64 index into ``self.paths``.

    Determinism: with ``deterministic=True`` the augmentation for item *i* in
    epoch *e* is a pure function of ``(seed, e, i)``, which makes a resumed run
    reproduce the original one. Call :meth:`set_epoch` before each epoch, and
    note that with ``persistent_workers=True`` the mutation will not reach
    already-forked workers -- the trainer therefore keeps that flag off when
    determinism is requested. With ``deterministic=False`` (the default for
    training) each read draws fresh entropy, which is both simpler and
    strictly better augmentation coverage.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        size: int = 448,
        augment: AugmentConfig | None = None,
        cache: TeacherCache | None = None,
        deterministic: bool = False,
        seed: int = 0,
        max_retries: int = 8,
    ) -> None:
        if not paths:
            raise ValueError("DistillationDataset was given no images")
        self.paths: list[Path] = list(paths)
        self.size = int(size)
        self.augment = augment if augment is not None else AugmentConfig()
        self.cache = cache
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.max_retries = int(max_retries)
        self.epoch = 0
        self.stats = DatasetStats()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.paths)

    def _rng(self, index: int) -> random.Random:
        if not self.deterministic:
            return random.Random()
        mixed = (self.seed * 1_000_003 + self.epoch * 9_176 + index) & 0xFFFF_FFFF
        return random.Random(mixed)

    def _load_one(self, index: int, rng: random.Random) -> dict[str, Tensor]:
        path = self.paths[index]
        image = load_image(path)
        self.stats.decoded += 1
        src_h, src_w = image.shape[-2:]

        geom = sample_geometry(src_h, src_w, self.augment, rng)
        clean = apply_geometry(image, geom, self.size, self.augment.interpolation)
        clean = clean.clamp_(0.0, 1.0)

        student = color_jitter(clean.clone(), self.augment, rng)

        if self.augment.exposure_log2 > 0.0:
            stops = rng.uniform(-self.augment.exposure_log2, self.augment.exposure_log2)
            exposure = float(2.0**stops)
        else:
            exposure = 1.0

        if self.cache is not None:
            target = self._load_target(path, geom, src_h, src_w)
        else:
            target = torch.empty(0, self.size, self.size, dtype=torch.float32)

        return {
            "student_rgb": student,
            "teacher_rgb": clean,
            "target": target,
            "exposure": torch.tensor(exposure, dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.int64),
        }

    def _load_target(
        self,
        path: Path,
        geom: GeometricParams,
        src_h: int,
        src_w: int,
    ) -> Tensor:
        assert self.cache is not None
        disparity, _ = self.cache.load(path)
        self.stats.cache_hits += 1
        cache_h, cache_w = disparity.shape[-2:]
        # The crop box lives in source-image pixels; replay it in cache pixels.
        local = rescale_geometry(geom, src_h, src_w, cache_h, cache_w)
        return apply_geometry(disparity, local, self.size, "bilinear")

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = self._rng(index)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._load_one(index, rng)
            except Exception as exc:  # corrupt file, missing cache entry, ...
                last_error = exc
                self.stats.failed += 1
                if attempt == 0:
                    warnings.warn(
                        f"failed to load sample {index} ({self.paths[index]}): "
                        f"{type(exc).__name__}: {exc}; substituting another sample",
                        stacklevel=2,
                    )
                # Deterministically walk to a different index rather than
                # returning zeros: a batch of zero images is a silent poison.
                index = (index + 1 + attempt * 7919) % len(self.paths)
                rng = self._rng(index)
        raise RuntimeError(
            f"gave up loading a sample after {self.max_retries} attempts; "
            f"last error: {last_error}"
        )


def split_paths(
    paths: Sequence[Path],
    val_fraction: float,
    seed: int = 0,
) -> tuple[list[Path], list[Path]]:
    """Deterministic train/val split.

    The split is a hash of the path, not a shuffle of the list, so adding
    images to the dataset does not move existing images across the boundary and
    invalidate the comparison with earlier runs.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if val_fraction == 0.0:
        return list(paths), []

    train: list[Path] = []
    val: list[Path] = []
    threshold = int(val_fraction * (1 << 32))
    for path in paths:
        digest = hashlib.sha1(f"{seed}:{path}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little")
        (val if bucket < threshold else train).append(path)
    if not train:
        raise ValueError("val_fraction left no training images")
    return train, val


# ---------------------------------------------------------------------------
# CLI: precompute a teacher cache
# ---------------------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m illumina.data",
        description=(
            "Precompute a teacher disparity cache for a directory of images. "
            "Do this once before training: the teacher costs roughly 40x a "
            "student training step, so caching turns a teacher-bound run into "
            "a student-bound one."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="directory of source images")
    parser.add_argument("--cache-dir", required=True, help="where to write entries")
    parser.add_argument(
        "--teacher",
        default="depth-anything-v2-small",
        help="teacher: depth-anything-v2-small | hf:<model-id> | synthetic",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="teacher compute dtype; fp16 halves the time at no visible cost",
    )
    parser.add_argument("--max-side", type=int, default=512, help="cached long side")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images")
    args = parser.parse_args(argv)

    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    device = torch.device(args.device)
    dtype = dtypes[args.dtype]
    if device.type == "cpu" and dtype is not torch.float32:
        print("cpu device: forcing fp32 for the teacher")
        dtype = torch.float32

    paths = find_images(args.root)
    if args.limit:
        paths = paths[: args.limit]
    teacher = build_teacher(args.teacher, device=device, dtype=dtype)
    cache = TeacherCache(args.cache_dir, teacher.name, args.max_side)
    precompute_teacher_cache(
        paths, teacher, cache, batch_size=args.batch_size, device=device
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
