"""PyTorch definition of IlluminaDepth-448.

This module is the Python twin of ``src/model/arch.ts`` and
``src/model/reference.ts``. It exists so that the network that is *trained*
here and the network that *runs* in the browser are provably the same
function, not two hand-written approximations of one idea.

Two rules keep them from drifting:

1. The architecture is a **declarative op table** (:func:`build_architecture`),
   a line-by-line port of ``buildArchitecture`` in ``src/model/arch.ts``. The
   modules, the parameter shapes, the export order and the FLOP report are all
   derived from that single table, exactly as the TypeScript side derives its
   kernels, its weight container and its cost report from its own copy.
2. Every learnable tensor is declared with the *logical* shape and the *exact*
   name that ``weightSpecs()`` in ``src/model/weights.ts`` produces --- for
   example ``e5a.dw_project.dw_weight`` of shape ``[C_in, k, k]``. Nothing is
   renamed or reshaped on the way out, so :func:`IlluminaDepth.fold_batchnorm`
   is a direct dump of the ``.idm`` container's contents.

Layout note: channels-last is a GPU packing detail, nothing more. Here every
tensor is ordinary NCHW and every weight is ordinary PyTorch order; the
swizzling into ``vec4`` groups happens in ``src/model/layout.ts`` at load time.

Semantics that are easy to get subtly wrong, and are therefore spelled out in
the layer implementations below:

* ``pw``      -- 1x1 conv, + bias, **then** add the residual, **then** activate.
* ``dwpw``    -- depthwise k x k (+bias, +midAct), then 1x1 (+bias), then
                 residual, then activation.
* ``lateral`` -- bilinear 2x upsample of the coarse tensor
                 (``align_corners=False``) **plus** a 1x1 projection of the
                 skip (+bias), then activation. The coarse tensor is not
                 projected: the previous level's refine step already emitted
                 the right width.
* ``head``    -- 1x1 conv to a single channel + scalar bias, then sigmoid.

Training uses a BatchNorm variant (``with_bn=True``); inference does not have
normalisation layers at all. :func:`IlluminaDepth.fold_batchnorm` collapses
each BatchNorm into the convolution in front of it and returns the
inference-shaped state dict that ``tools/export/export_idm.py`` writes.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "Activation",
    "TensorShape",
    "Architecture",
    "Op",
    "ConvOp",
    "PointwiseOp",
    "DepthwisePointwiseOp",
    "LateralOp",
    "HeadOp",
    "PreprocessOp",
    "BilateralUpOp",
    "build_architecture",
    "weight_specs",
    "total_parameters",
    "cost_of",
    "apply_activation",
    "srgb_to_linear",
    "preprocess",
    "IlluminaDepth",
    "build_model",
]

# ---------------------------------------------------------------------------
# Architecture description (port of src/model/arch.ts)
# ---------------------------------------------------------------------------

Activation = Literal["linear", "relu", "relu6", "hardswish"]


@dataclass(frozen=True)
class TensorShape:
    """Shape of an activation tensor. ``c`` is always a multiple of four."""

    h: int
    w: int
    c: int


@dataclass(frozen=True)
class PreprocessOp:
    """Image -> network input. Not a learnable op; see :func:`preprocess`."""

    name: str
    out: str
    kind: str = "preprocess"


@dataclass(frozen=True)
class ConvOp:
    """Dense k x k convolution, symmetric zero padding ``(k - 1) // 2``."""

    name: str
    inp: str
    out: str
    k: int
    stride: int
    act: Activation
    kind: str = "conv"


@dataclass(frozen=True)
class PointwiseOp:
    """1x1 convolution + bias (+ residual) + activation."""

    name: str
    inp: str
    out: str
    act: Activation
    residual: str | None = None
    kind: str = "pw"


@dataclass(frozen=True)
class DepthwisePointwiseOp:
    """Fused depthwise k x k (+bias +midAct) followed by 1x1 (+bias +act)."""

    name: str
    inp: str
    out: str
    k: int
    stride: int
    mid_act: Activation
    act: Activation
    residual: str | None = None
    kind: str = "dwpw"


@dataclass(frozen=True)
class LateralOp:
    """FPN fusion: ``act(up2(coarse) + W . skip + b)``."""

    name: str
    coarse: str
    skip: str
    out: str
    act: Activation
    kind: str = "lateral"


@dataclass(frozen=True)
class HeadOp:
    """1 x C convolution to a single channel + scalar bias, then sigmoid."""

    name: str
    inp: str
    out: str
    kind: str = "head"


@dataclass(frozen=True)
class BilateralUpOp:
    """Edge-aware upsample to full resolution. Inference-only, no parameters."""

    name: str
    inp: str
    out: str
    kind: str = "bilateralUp"


Op = (
    PreprocessOp
    | ConvOp
    | PointwiseOp
    | DepthwisePointwiseOp
    | LateralOp
    | HeadOp
    | BilateralUpOp
)


@dataclass(frozen=True)
class Architecture:
    """The whole network, as data."""

    name: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    tensors: dict[str, TensorShape]
    ops: list[Op] = field(default_factory=list)

    def shape(self, name: str) -> TensorShape:
        try:
            return self.tensors[name]
        except KeyError as exc:  # pragma: no cover - programming error
            raise KeyError(f"unknown tensor {name}") from exc

    def channels(self, name: str) -> int:
        return self.shape(name).c


def build_architecture(input_size: int = 448) -> Architecture:
    """Builds the architecture at a given square input resolution.

    A line-by-line port of ``buildArchitecture`` in ``src/model/arch.ts``. The
    resolution is a parameter so the parity tests can instantiate a
    geometrically identical but much smaller network (448 / 7 = 64) and compare
    a full forward pass against the CPU reference in seconds.

    ``input_size`` must be a multiple of 32: the encoder downsamples five times.
    """
    if input_size % 32 != 0:
        raise ValueError(f"input_size {input_size} must be a multiple of 32")

    tensors: dict[str, TensorShape] = {}
    ops: list[Op] = []

    def t(name: str, h: int, w: int, c: int) -> str:
        if c % 4 != 0:
            raise ValueError(f"tensor {name}: channel count {c} is not a multiple of 4")
        tensors[name] = TensorShape(h, w, c)
        return name

    s = input_size

    def r(level: int) -> int:
        return s >> level

    # Normalised RGB input; the alpha channel carries luminance for later reuse.
    x0 = t("input", s, s, 4)
    ops.append(PreprocessOp(name="preprocess", out=x0))

    # --- Stem: S -> S/2 ---------------------------------------------------
    stem = t("stem", r(1), r(1), 16)
    ops.append(ConvOp(name="stem", inp=x0, out=stem, k=3, stride=2, act="hardswish"))

    # A depthwise-separable block with no expansion, as in MobileNetV2's first
    # bottleneck. Produces the highest-resolution skip.
    s0 = t("s0", r(1), r(1), 24)
    ops.append(
        DepthwisePointwiseOp(
            name="e1", inp=stem, out=s0, k=3, stride=1, mid_act="relu6", act="linear"
        )
    )

    def down_block(
        prefix: str, inp: str, h: int, w: int, expand: int, out: int, k: int
    ) -> str:
        """Downsampling block: depthwise s2 -> expand, then project.

        The depthwise runs *first*, at stride 2, so the expensive channel
        expansion happens at the lower resolution -- a 4x saving versus
        MobileNetV2's ordering, which expands before downsampling.
        """
        mid = t(f"{prefix}.mid", h, w, expand)
        ops.append(
            DepthwisePointwiseOp(
                name=f"{prefix}.dw_expand",
                inp=inp,
                out=mid,
                k=k,
                stride=2,
                mid_act="linear",
                act="hardswish",
            )
        )
        o = t(f"{prefix}.out", h, w, out)
        ops.append(PointwiseOp(name=f"{prefix}.project", inp=mid, out=o, act="linear"))
        return o

    def ir_block(
        prefix: str, inp: str, h: int, w: int, channels: int, expand: int, k: int
    ) -> str:
        """Residual inverted bottleneck: 1x1 expand -> depthwise k x k -> 1x1."""
        mid = t(f"{prefix}.mid", h, w, expand)
        ops.append(
            PointwiseOp(name=f"{prefix}.expand", inp=inp, out=mid, act="hardswish")
        )
        o = t(f"{prefix}.out", h, w, channels)
        ops.append(
            DepthwisePointwiseOp(
                name=f"{prefix}.dw_project",
                inp=mid,
                out=o,
                k=k,
                stride=1,
                mid_act="hardswish",
                act="linear",
                residual=inp,
            )
        )
        return o

    # --- Stage 1: S/2 -> S/4, 32 channels ---------------------------------
    x = down_block("e2", s0, r(2), r(2), 96, 32, 3)
    x = ir_block("e3", x, r(2), r(2), 32, 128, 3)
    s1 = x

    # --- Stage 2: S/4 -> S/8, 56 channels ---------------------------------
    x = down_block("e4", s1, r(3), r(3), 128, 56, 5)
    x = ir_block("e5a", x, r(3), r(3), 56, 224, 5)
    x = ir_block("e5b", x, r(3), r(3), 56, 224, 5)
    s2 = x

    # --- Stage 3: S/8 -> S/16, 104 channels -------------------------------
    x = down_block("e6", s2, r(4), r(4), 224, 104, 5)
    x = ir_block("e7a", x, r(4), r(4), 104, 416, 5)
    x = ir_block("e7b", x, r(4), r(4), 104, 416, 5)
    x = ir_block("e7c", x, r(4), r(4), 104, 416, 5)
    s3 = x

    # --- Stage 4: S/16 -> S/32, 176 channels (bottleneck) -----------------
    x = down_block("e8", s3, r(5), r(5), 416, 176, 5)
    x = ir_block("e9a", x, r(5), r(5), 176, 704, 5)
    x = ir_block("e9b", x, r(5), r(5), 176, 704, 5)
    x = ir_block("e9c", x, r(5), r(5), 176, 704, 5)
    bottleneck = x

    # --- Decoder ----------------------------------------------------------
    d4 = t("d4", r(5), r(5), 96)
    ops.append(
        PointwiseOp(name="d4.lateral", inp=bottleneck, out=d4, act="hardswish")
    )

    def decoder_level(
        prefix: str,
        coarse: str,
        skip: str,
        h: int,
        w: int,
        channels: int,
        next_channels: int,
    ) -> str:
        """One FPN level: fuse, then refine into the width the level above wants."""
        fused = t(f"{prefix}.fused", h, w, channels)
        ops.append(
            LateralOp(
                name=f"{prefix}.lateral",
                coarse=coarse,
                skip=skip,
                out=fused,
                act="linear",
            )
        )
        o = t(f"{prefix}.out", h, w, next_channels)
        ops.append(
            DepthwisePointwiseOp(
                name=f"{prefix}.refine",
                inp=fused,
                out=o,
                k=3,
                stride=1,
                mid_act="hardswish",
                act="hardswish",
                # A residual is only meaningful when the width is unchanged.
                residual=fused if channels == next_channels else None,
            )
        )
        return o

    d3 = decoder_level("d3", d4, s3, r(4), r(4), 96, 64)
    d2 = decoder_level("d2", d3, s2, r(3), r(3), 64, 48)
    d1 = decoder_level("d1", d2, s1, r(2), r(2), 48, 32)
    d0 = decoder_level("d0", d1, s0, r(1), r(1), 32, 32)

    # --- Head -------------------------------------------------------------
    depth_low = t("depth_low", r(1), r(1), 4)
    ops.append(HeadOp(name="head", inp=d0, out=depth_low))

    depth_full = t("depth", s, s, 4)
    ops.append(BilateralUpOp(name="upsample", inp=depth_low, out=depth_full))

    return Architecture(
        name="IlluminaDepth-448",
        input_size=s,
        # ImageNet statistics, applied in linear-light space.
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        tensors=tensors,
        ops=ops,
    )


# ---------------------------------------------------------------------------
# Weight table (port of weightSpecs() in src/model/weights.ts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightSpec:
    """One learnable tensor: its exported name and its logical shape."""

    name: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def weight_specs(arch: Architecture) -> list[WeightSpec]:
    """Every weight tensor the architecture needs, in ``.idm`` blob order.

    Must stay byte-for-byte consistent with ``weightSpecs()`` in
    ``src/model/weights.ts``: same names, same shapes, same order.
    """
    out: list[WeightSpec] = []
    c = arch.channels

    for op in arch.ops:
        if isinstance(op, ConvOp):
            out.append(WeightSpec(f"{op.name}.weight", (c(op.out), c(op.inp), op.k, op.k)))
            out.append(WeightSpec(f"{op.name}.bias", (c(op.out),)))
        elif isinstance(op, PointwiseOp):
            out.append(WeightSpec(f"{op.name}.weight", (c(op.out), c(op.inp))))
            out.append(WeightSpec(f"{op.name}.bias", (c(op.out),)))
        elif isinstance(op, DepthwisePointwiseOp):
            out.append(WeightSpec(f"{op.name}.dw_weight", (c(op.inp), op.k, op.k)))
            out.append(WeightSpec(f"{op.name}.dw_bias", (c(op.inp),)))
            out.append(WeightSpec(f"{op.name}.pw_weight", (c(op.out), c(op.inp))))
            out.append(WeightSpec(f"{op.name}.pw_bias", (c(op.out),)))
        elif isinstance(op, LateralOp):
            out.append(WeightSpec(f"{op.name}.weight", (c(op.out), c(op.skip))))
            out.append(WeightSpec(f"{op.name}.bias", (c(op.out),)))
        elif isinstance(op, HeadOp):
            out.append(WeightSpec(f"{op.name}.weight", (1, c(op.inp))))
            out.append(WeightSpec(f"{op.name}.bias", (1,)))
        # preprocess and bilateralUp carry no parameters.
    return out


def total_parameters(arch: Architecture) -> int:
    """Total number of learnable scalars in the inference graph."""
    return sum(spec.numel for spec in weight_specs(arch))


@dataclass(frozen=True)
class OpCost:
    """Multiply-accumulates and parameters attributable to one op."""

    name: str
    kind: str
    macs: int
    params: int
    out_shape: TensorShape


def cost_of(arch: Architecture) -> tuple[list[OpCost], int, int]:
    """Exact MAC and parameter counts. Mirrors ``costOf`` in ``arch.ts``.

    Returns ``(per_op, total_macs, total_params)``. FLOPs are ``2 * macs``.
    """
    costs: list[OpCost] = []
    for op in arch.ops:
        if isinstance(op, ConvOp):
            o, i = arch.shape(op.out), arch.shape(op.inp)
            macs = o.h * o.w * o.c * i.c * op.k * op.k
            params = o.c * i.c * op.k * op.k + o.c
        elif isinstance(op, PointwiseOp):
            o, i = arch.shape(op.out), arch.shape(op.inp)
            macs = o.h * o.w * o.c * i.c
            params = o.c * i.c + o.c
        elif isinstance(op, DepthwisePointwiseOp):
            o, i = arch.shape(op.out), arch.shape(op.inp)
            # The depthwise runs at the *output* resolution, over input channels.
            macs = o.h * o.w * i.c * op.k * op.k + o.h * o.w * o.c * i.c
            params = i.c * op.k * op.k + i.c + o.c * i.c + o.c
        elif isinstance(op, LateralOp):
            o, sk, co = arch.shape(op.out), arch.shape(op.skip), arch.shape(op.coarse)
            if co.c != o.c:
                raise ValueError(
                    f"{op.name}: coarse tensor has {co.c} channels but the output "
                    f"has {o.c}; the previous level's refine step must emit the "
                    "matching width"
                )
            macs = o.h * o.w * o.c * sk.c
            params = o.c * sk.c + o.c
        elif isinstance(op, HeadOp):
            o, i = arch.shape(op.out), arch.shape(op.inp)
            macs = o.h * o.w * i.c
            params = i.c + 1
        else:  # preprocess, bilateralUp
            o = arch.shape(op.out)
            macs = 0
            params = 0
        costs.append(OpCost(op.name, op.kind, macs, params, o))

    return costs, sum(c.macs for c in costs), sum(c.params for c in costs)


# ---------------------------------------------------------------------------
# Activations (must match applyActivation in src/model/layout.ts exactly)
# ---------------------------------------------------------------------------


def apply_activation(act: Activation, x: Tensor) -> Tensor:
    """Elementwise activation, identical to ``applyActivation`` in layout.ts.

    ``hardswish`` is written out rather than delegated to ``nn.Hardswish`` so
    the formula is checkable by eye; the two are numerically identical:
    ``x * clamp(x + 3, 0, 6) / 6``.
    """
    if act == "linear":
        return x
    if act == "relu":
        return torch.clamp(x, min=0.0)
    if act == "relu6":
        return torch.clamp(x, min=0.0, max=6.0)
    if act == "hardswish":
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0
    raise ValueError(f"unknown activation {act!r}")


# ---------------------------------------------------------------------------
# Preprocessing (must match refPreprocess in src/model/reference.ts)
# ---------------------------------------------------------------------------


def srgb_to_linear(c: Tensor) -> Tensor:
    """Exact sRGB electro-optical transfer function (not a gamma-2.2 approximation).

    ``c <= 0.04045 -> c / 12.92`` else ``((c + 0.055) / 1.055) ** 2.4``.

    ``torch.where`` evaluates both branches, so the power branch's base is
    clamped at zero to keep negative inputs from producing NaN; for in-range
    inputs the clamp is a no-op.
    """
    lo = c / 12.92
    hi = torch.pow(torch.clamp((c + 0.055) / 1.055, min=0.0), 2.4)
    return torch.where(c <= 0.04045, lo, hi)


def luminance_of(linear_rgb: Tensor) -> Tensor:
    """Rec. 709 luminance of a linear-light RGB tensor ``[N, 3, H, W]``."""
    r, g, b = linear_rgb[:, 0], linear_rgb[:, 1], linear_rgb[:, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def preprocess(
    rgb: Tensor,
    *,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    decode_srgb: bool = True,
    exposure: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Turns an sRGB image into the network's 4-channel input tensor.

    Port of ``refPreprocess`` in ``src/model/reference.ts``. ``rgb`` is
    ``[N, 3, H, W]`` with components in ``[0, 1]``.

    The order is load-bearing:

    1. sRGB -> linear light (exact piecewise curve),
    2. multiply by ``exposure``,
    3. normalise with the ImageNet mean/std --- **in linear light**, not in
       sRGB. This is unusual but it is what the shader does, and a student
       trained on sRGB-normalised inputs would be systematically wrong.

    Channel 3 of the input carries the *unnormalised* linear luminance, which
    the lighting pass and the joint-bilateral upsample reuse later.

    Returns ``(network_input[N, 4, H, W], scene_color[N, 3, H, W])`` where
    ``scene_color`` is the exposure-scaled linear-light image.
    """
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"preprocess expects [N, 3, H, W], got {tuple(rgb.shape)}")

    linear = (srgb_to_linear(rgb) if decode_srgb else rgb) * exposure
    m = torch.as_tensor(mean, dtype=linear.dtype, device=linear.device).view(1, 3, 1, 1)
    s = torch.as_tensor(std, dtype=linear.dtype, device=linear.device).view(1, 3, 1, 1)
    normalised = (linear - m) / s
    luma = luminance_of(linear).unsqueeze(1)
    return torch.cat([normalised, luma], dim=1), linear


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
#
# Every layer declares its parameters with the *exported* name and the
# *exported* shape, and reshapes them on the fly for ``F.conv2d``. Declaring
# them as ``nn.Conv2d`` instead would give 1x1 kernels a spurious trailing
# ``[1, 1]`` in the state dict and force a rename at export time; doing it this
# way means the training state dict and the ``.idm`` container agree by
# construction.
#
# BatchNorm folding
# -----------------
# For a convolution followed by BatchNorm in inference mode,
#
#     y = gamma * (conv(x, w, b) - mean) / sqrt(var + eps) + beta
#
# which, writing ``scale = gamma / sqrt(var + eps)``, is exactly another
# convolution:
#
#     w' = w * scale                      (broadcast over the output-channel axis)
#     b' = (b - mean) * scale + beta
#
# ``fold_batchnorm`` applies precisely that, using the running statistics, so
# the folded network is bit-comparable to the BatchNorm one in ``eval()`` mode.


def _make_bn(channels: int, enabled: bool) -> nn.BatchNorm2d | None:
    return nn.BatchNorm2d(channels) if enabled else None


def _fold(
    weight: Tensor, bias: Tensor, bn: nn.BatchNorm2d | None, weight_dims: int
) -> tuple[Tensor, Tensor]:
    """Folds ``bn`` into ``(weight, bias)``; returns them unchanged if ``bn`` is None.

    ``weight_dims`` is the rank of the weight tensor so the per-output-channel
    scale can be broadcast along the right number of trailing axes.
    """
    if bn is None:
        return weight.detach().clone(), bias.detach().clone()
    if bn.running_mean is None or bn.running_var is None:
        raise RuntimeError("fold_batchnorm requires track_running_stats=True")
    gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_var)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_var)
    scale = gamma / torch.sqrt(bn.running_var + bn.eps)
    view = (-1,) + (1,) * (weight_dims - 1)
    folded_w = weight * scale.view(*view)
    folded_b = (bias - bn.running_mean) * scale + beta
    return folded_w.detach().clone(), folded_b.detach().clone()


class ConvLayer(nn.Module):
    """``conv`` op: dense k x k + bias (+ BN) + activation."""

    def __init__(
        self, in_c: int, out_c: int, k: int, stride: int, act: Activation, with_bn: bool
    ) -> None:
        super().__init__()
        self.k = k
        self.stride = stride
        self.padding = (k - 1) // 2
        self.act: Activation = act
        self.weight = nn.Parameter(torch.empty(out_c, in_c, k, k))
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.bn = _make_bn(out_c, with_bn)

    def forward(self, x: Tensor) -> Tensor:
        y = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        if self.bn is not None:
            y = self.bn(y)
        return apply_activation(self.act, y)

    def folded(self) -> dict[str, Tensor]:
        w, b = _fold(self.weight, self.bias, self.bn, 4)
        return {"weight": w, "bias": b}


class PointwiseLayer(nn.Module):
    """``pw`` op: 1x1 + bias (+ BN), then residual, then activation.

    The residual is added *before* the activation. Adding it afterwards would
    change the function, so the order is asserted by the parity test rather
    than assumed.
    """

    def __init__(self, in_c: int, out_c: int, act: Activation, with_bn: bool) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.act: Activation = act
        self.weight = nn.Parameter(torch.empty(out_c, in_c))
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.bn = _make_bn(out_c, with_bn)

    def forward(self, x: Tensor, residual: Tensor | None = None) -> Tensor:
        y = F.conv2d(x, self.weight.view(self.out_c, self.in_c, 1, 1), self.bias)
        if self.bn is not None:
            y = self.bn(y)
        if residual is not None:
            y = y + residual
        return apply_activation(self.act, y)

    def folded(self) -> dict[str, Tensor]:
        w, b = _fold(self.weight, self.bias, self.bn, 2)
        return {"weight": w, "bias": b}


class DepthwisePointwiseLayer(nn.Module):
    """``dwpw`` op: depthwise k x k (+bias +midAct) then 1x1 (+bias) (+res) +act."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int,
        stride: int,
        mid_act: Activation,
        act: Activation,
        with_bn: bool,
    ) -> None:
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.stride = stride
        self.padding = (k - 1) // 2
        self.mid_act: Activation = mid_act
        self.act: Activation = act
        # Exported shape is [C, k, k]; F.conv2d wants [C, 1, k, k] with groups=C.
        self.dw_weight = nn.Parameter(torch.empty(in_c, k, k))
        self.dw_bias = nn.Parameter(torch.zeros(in_c))
        self.pw_weight = nn.Parameter(torch.empty(out_c, in_c))
        self.pw_bias = nn.Parameter(torch.zeros(out_c))
        self.bn_dw = _make_bn(in_c, with_bn)
        self.bn_pw = _make_bn(out_c, with_bn)

    def forward(self, x: Tensor, residual: Tensor | None = None) -> Tensor:
        mid = F.conv2d(
            x,
            self.dw_weight.unsqueeze(1),
            self.dw_bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_c,
        )
        if self.bn_dw is not None:
            mid = self.bn_dw(mid)
        mid = apply_activation(self.mid_act, mid)

        y = F.conv2d(mid, self.pw_weight.view(self.out_c, self.in_c, 1, 1), self.pw_bias)
        if self.bn_pw is not None:
            y = self.bn_pw(y)
        if residual is not None:
            y = y + residual
        return apply_activation(self.act, y)

    def folded(self) -> dict[str, Tensor]:
        dw_w, dw_b = _fold(self.dw_weight, self.dw_bias, self.bn_dw, 3)
        pw_w, pw_b = _fold(self.pw_weight, self.pw_bias, self.bn_pw, 2)
        return {
            "dw_weight": dw_w,
            "dw_bias": dw_b,
            "pw_weight": pw_w,
            "pw_bias": pw_b,
        }


class LateralLayer(nn.Module):
    """``lateral`` op: 2x bilinear upsample of ``coarse`` + 1x1 projection of ``skip``.

    ``coarse`` is deliberately *not* projected: the previous decoder level's
    refine step already emitted this level's channel width, which saves a
    full-resolution 1x1 convolution per level.
    """

    def __init__(self, skip_c: int, out_c: int, act: Activation, with_bn: bool) -> None:
        super().__init__()
        self.skip_c = skip_c
        self.out_c = out_c
        self.act: Activation = act
        self.weight = nn.Parameter(torch.empty(out_c, skip_c))
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.bn = _make_bn(out_c, with_bn)

    def forward(self, coarse: Tensor, skip: Tensor) -> Tensor:
        if coarse.shape[1] != self.out_c:
            raise ValueError(
                f"lateral: coarse has {coarse.shape[1]} channels, output has {self.out_c}"
            )
        target = (coarse.shape[2] * 2, coarse.shape[3] * 2)
        if tuple(skip.shape[2:]) != target:
            raise ValueError(
                f"lateral: upsampled coarse is {target}, skip is {tuple(skip.shape[2:])}"
            )
        # align_corners=False matches refUpsample2: src = (i + 0.5) / 2 - 0.5.
        up = F.interpolate(coarse, size=target, mode="bilinear", align_corners=False)
        proj = F.conv2d(skip, self.weight.view(self.out_c, self.skip_c, 1, 1), self.bias)
        if self.bn is not None:
            proj = self.bn(proj)
        return apply_activation(self.act, up + proj)

    def folded(self) -> dict[str, Tensor]:
        w, b = _fold(self.weight, self.bias, self.bn, 2)
        return {"weight": w, "bias": b}


class HeadLayer(nn.Module):
    """``head`` op: 1 x C convolution to one channel + scalar bias, then sigmoid.

    No BatchNorm here, deliberately. This layer's scale and shift *are* the
    output calibration; normalising them per batch would fight the sigmoid and
    the scale-and-shift-invariant loss, and there is no BN to fold at export.

    The CPU reference broadcasts the result across four channels so it shares
    the packed ``vec4`` layout; PyTorch keeps it as a single channel.
    """

    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.in_c = in_c
        self.weight = nn.Parameter(torch.empty(1, in_c))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        y = F.conv2d(x, self.weight.view(1, self.in_c, 1, 1), self.bias)
        return torch.sigmoid(y)

    def folded(self) -> dict[str, Tensor]:
        return {
            "weight": self.weight.detach().clone(),
            "bias": self.bias.detach().clone(),
        }


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


def _register(root: nn.Module, dotted: str, module: nn.Module) -> None:
    """Registers ``module`` at a dotted path, creating containers as needed.

    ``nn.Module.add_module`` rejects names containing a dot, but the op names in
    the architecture table (``e5a.dw_project``) are dotted. Splitting them into
    nested plain ``nn.Module`` containers reproduces the dotted name in the
    state dict, which is exactly what ``weightSpecs()`` expects.
    """
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        child = getattr(parent, part, None)
        if child is None:
            child = nn.Module()
            parent.add_module(part, child)
        elif not isinstance(child, nn.Module):  # pragma: no cover
            raise TypeError(f"{part} is already bound to a non-module")
        parent = child
    parent.add_module(parts[-1], module)


class IlluminaDepth(nn.Module):
    """IlluminaDepth-448: 448x448 RGB in, relative inverse depth (disparity) out.

    ``forward`` takes the preprocessed 4-channel tensor (see :func:`preprocess`)
    and returns disparity in ``[0, 1]`` at half the input resolution --- the
    resolution the head produces. The edge-aware upsample to full resolution is
    an inference-time, guide-dependent op (``bilateralUp``) and is not part of
    the trainable graph; train against the half-resolution output, or bilinearly
    upsample it, and let the shader do the edge-aware version at runtime.

    ``with_bn=True`` inserts a BatchNorm after every convolution except the
    head. Call :meth:`fold_batchnorm` to get the inference-shaped state dict.
    """

    def __init__(self, arch: Architecture, with_bn: bool = True) -> None:
        super().__init__()
        self.arch = arch
        self.with_bn = with_bn
        self._layers: dict[str, nn.Module] = {}

        for op in arch.ops:
            layer: nn.Module | None = None
            if isinstance(op, ConvOp):
                layer = ConvLayer(
                    arch.channels(op.inp),
                    arch.channels(op.out),
                    op.k,
                    op.stride,
                    op.act,
                    with_bn,
                )
            elif isinstance(op, PointwiseOp):
                layer = PointwiseLayer(
                    arch.channels(op.inp), arch.channels(op.out), op.act, with_bn
                )
            elif isinstance(op, DepthwisePointwiseOp):
                layer = DepthwisePointwiseLayer(
                    arch.channels(op.inp),
                    arch.channels(op.out),
                    op.k,
                    op.stride,
                    op.mid_act,
                    op.act,
                    with_bn,
                )
            elif isinstance(op, LateralOp):
                layer = LateralLayer(
                    arch.channels(op.skip), arch.channels(op.out), op.act, with_bn
                )
            elif isinstance(op, HeadOp):
                layer = HeadLayer(arch.channels(op.inp))
            if layer is not None:
                _register(self, op.name, layer)
                self._layers[op.name] = layer

        self.reset_parameters()

    # -- initialisation ----------------------------------------------------

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        """Kaiming-normal initialisation, matching ``synthesizeWeights`` in weights.ts.

        ``std = sqrt(2 / fan_in)`` with ``fan_in`` counting everything but the
        leading (output) axis; for a depthwise kernel of shape ``[C, k, k]`` the
        groups make ``fan_in`` just ``k * k``. Biases start at zero, as
        PyTorch's convolution default does. The head is scaled down by 20x so an
        untrained network emits a mid-grey disparity rather than saturated noise.
        """
        for name, spec_shape, param in self._named_weight_parameters():
            if name.endswith("bias"):
                nn.init.zeros_(param)
                continue
            if name.endswith("dw_weight"):
                fan_in = spec_shape[1] * spec_shape[2]
            else:
                fan_in = math.prod(spec_shape[1:]) if len(spec_shape) > 1 else 1
            std = math.sqrt(2.0 / max(fan_in, 1))
            if name.startswith("head"):
                std *= 0.05
            with torch.no_grad():
                param.normal_(0.0, std, generator=generator)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_parameters()
                module.reset_running_stats()

    def _named_weight_parameters(self) -> list[tuple[str, tuple[int, ...], nn.Parameter]]:
        """Pairs every :func:`weight_specs` entry with its live ``nn.Parameter``."""
        params = dict(self.named_parameters())
        out: list[tuple[str, tuple[int, ...], nn.Parameter]] = []
        for spec in weight_specs(self.arch):
            param = params.get(spec.name)
            if param is None:
                raise KeyError(f"model is missing parameter {spec.name}")
            if tuple(param.shape) != spec.shape:
                raise ValueError(
                    f"{spec.name}: model has shape {tuple(param.shape)}, "
                    f"weightSpecs wants {spec.shape}"
                )
            out.append((spec.name, spec.shape, param))
        return out

    # -- forward -----------------------------------------------------------

    def forward(
        self, x: Tensor, return_intermediates: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Runs the graph.

        ``x`` is ``[N, 4, H, W]`` --- the output of :func:`preprocess`, not a
        raw image. Returns ``[N, 1, H/2, W/2]`` disparity in ``[0, 1]``.

        With ``return_intermediates=True`` it also returns every activation
        keyed by the architecture's tensor names, so ``verify_parity.py`` can
        bisect a divergence down to the first op that disagrees with the
        TypeScript reference.
        """
        if x.dim() != 4 or x.shape[1] != self.arch.channels("input"):
            raise ValueError(
                f"expected [N, {self.arch.channels('input')}, H, W], "
                f"got {tuple(x.shape)}"
            )

        values: dict[str, Tensor] = {"input": x}

        def get(name: str) -> Tensor:
            try:
                return values[name]
            except KeyError as exc:  # pragma: no cover - table is static
                raise KeyError(f"{name} has not been produced yet") from exc

        for op in self.arch.ops:
            if isinstance(op, (PreprocessOp, BilateralUpOp)):
                # preprocess happens outside the module; bilateralUp needs the
                # guide image and is an inference-only shader pass.
                continue
            layer = self._layers[op.name]
            if isinstance(op, ConvOp):
                values[op.out] = layer(get(op.inp))
            elif isinstance(op, (PointwiseOp, DepthwisePointwiseOp)):
                residual = get(op.residual) if op.residual is not None else None
                values[op.out] = layer(get(op.inp), residual)
            elif isinstance(op, LateralOp):
                values[op.out] = layer(get(op.coarse), get(op.skip))
            elif isinstance(op, HeadOp):
                values[op.out] = layer(get(op.inp))

        depth = values["depth_low"]
        if return_intermediates:
            return depth, values
        return depth

    # -- export ------------------------------------------------------------

    def fold_batchnorm(self) -> "OrderedDict[str, Tensor]":
        """Folds every BatchNorm into its convolution and returns the export dict.

        The maths, for one output channel, with running statistics ``mean`` and
        ``var`` and BatchNorm affine parameters ``gamma`` and ``beta``:

            w' = w * gamma / sqrt(var + eps)
            b' = (b - mean) * gamma / sqrt(var + eps) + beta

        The result is keyed and ordered exactly like ``weightSpecs()``, with
        every tensor detached, on the CPU, in float32 --- ready for
        ``tools/export/export_idm.py`` to write straight into the ``.idm`` blob.

        Folding uses the *running* statistics, so a model trained with
        ``track_running_stats=False`` cannot be folded; and a folded model
        matches the BatchNorm model in ``eval()`` mode, not in ``train()`` mode.
        """
        folded: dict[str, Tensor] = {}
        for op in self.arch.ops:
            layer = self._layers.get(op.name)
            if layer is None:
                continue
            for suffix, tensor in layer.folded().items():  # type: ignore[operator]
                folded[f"{op.name}.{suffix}"] = tensor.to(
                    device="cpu", dtype=torch.float32
                )

        out: "OrderedDict[str, Tensor]" = OrderedDict()
        for spec in weight_specs(self.arch):
            tensor = folded.pop(spec.name, None)
            if tensor is None:
                raise KeyError(f"fold_batchnorm produced no tensor for {spec.name}")
            if tuple(tensor.shape) != spec.shape:
                raise ValueError(
                    f"{spec.name}: folded shape {tuple(tensor.shape)} != "
                    f"expected {spec.shape}"
                )
            out[spec.name] = tensor.contiguous()
        if folded:
            raise ValueError(
                f"fold_batchnorm produced unexpected tensors: {sorted(folded)}"
            )
        return out

    def load_folded_state_dict(self, state: dict[str, Tensor]) -> None:
        """Loads an inference-shaped (BatchNorm-free) state dict into this model.

        Only valid on a ``with_bn=False`` model: folding is not invertible.
        """
        if self.with_bn:
            raise RuntimeError(
                "load_folded_state_dict needs a with_bn=False model; a folded "
                "checkpoint has no BatchNorm statistics to restore"
            )
        missing, unexpected = self.load_state_dict(state, strict=True)  # type: ignore[misc]
        del missing, unexpected

    def num_parameters(self) -> int:
        """Learnable scalars in the *inference* graph (BatchNorm excluded)."""
        return total_parameters(self.arch)


def build_model(input_size: int = 448, with_bn: bool = True) -> IlluminaDepth:
    """Builds IlluminaDepth at ``input_size`` (a multiple of 32).

    ``with_bn=True`` is the training configuration; ``with_bn=False`` is the
    inference graph that the ``.idm`` container describes.
    """
    return IlluminaDepth(build_architecture(input_size), with_bn=with_bn)


if __name__ == "__main__":  # pragma: no cover - a convenience report
    architecture = build_architecture(448)
    per_op, macs, params = cost_of(architecture)
    print(f"{architecture.name} - {architecture.input_size}x{architecture.input_size}")
    for c in per_op:
        print(
            f"{c.name:<20}{c.kind:<13}"
            f"{c.out_shape.h}x{c.out_shape.w}x{c.out_shape.c:<8}"
            f"{c.macs * 2e-6:>10.1f}{c.params:>10}"
        )
    print(f"GFLOP / frame : {macs * 2e-9:.3f}")
    print(f"parameters    : {params / 1e6:.2f} M")
    print(f"weight tensors: {len(weight_specs(architecture))}")
