"""Distillation trainer for IlluminaDepth-448.

What this script does
---------------------
Trains the 1.39 M parameter student in ``illumina/model.py`` to reproduce the
relative inverse depth (disparity) of a frozen teacher -- Depth Anything V2
Small by default -- over a directory of ordinary photographs, using the MiDaS
scale-and-shift-invariant loss plus a multi-scale gradient-matching term:

    L = L_ssi(trimmed) + 0.5 * L_reg

Why distillation rather than supervised training: the student is roughly two
orders of magnitude smaller than any published depth model, and the public
metric-depth datasets are small, sensor-specific and mutually inconsistent. A
frozen teacher run over a large pile of unlabelled images gives dense,
self-consistent targets at whatever volume the disk can hold. This is the same
recipe the teacher's own authors use to make their small variants.

What it provides
----------------
AdamW with decoupled weight decay (never applied to biases or normalisation
parameters), a cosine schedule with linear warmup, mixed precision, gradient
clipping, gradient accumulation, an exponential moving average of the weights,
atomic checkpointing with exact resume, and periodic validation reporting
AbsRel and delta1 after a least-squares affine alignment of the prediction to
the target.

Checkpoint schema (also read by ``tools/export/export_idm.py``)
---------------------------------------------------------------
``{"format": "illumina-ckpt/1", "arch": "IlluminaDepth-448", "input_size": int,
   "with_bn": bool, "step": int, "epoch": int, "model": state_dict,
   "ema": state_dict | None, "optimizer": ..., "scheduler": ..., "scaler": ...,
   "metrics": dict, "best_metric": float, "args": dict, "teacher": str}``

Export the ``ema`` weights when they are present and ``model`` otherwise.

The ``model`` and ``ema`` state dicts are the *training* (BatchNorm) shape.
Folding happens at export time, not here, so that a checkpoint can always be
resumed.

Honest cost
-----------
Run ``--help`` or start a run: the script prints a cost estimate derived from
the actual dataset size and settings before it does any work. The short
version, for 448x448 and a precomputed teacher cache:

  * building the cache costs about 90 GFLOP per source image (roughly
    3-5 GPU-hours per 300 k images on an A100), paid once;
  * a student training step costs about 6.6 GFLOP per sample, so training is
    input-pipeline bound long before it is GPU bound;
  * a usable model needs on the order of 1-3 M samples seen, i.e. 15-30 A100
    hours on top of the cache. Fewer than ~50 k distinct source images gives a
    model that memorises the teacher's quirks on those images and generalises
    poorly.

Requires torch >= 2.4 (for the ``torch.amp`` device-first API).

Examples
--------
    # 0. one-off: build the teacher cache (strongly recommended)
    python -m illumina.data --root /data/images --cache-dir /data/cache

    # 1. train
    python tools/train/train.py --data-root /data/images \\
        --teacher-cache /data/cache --out runs/v1 --epochs 30 --batch-size 32

    # 2. resume
    python tools/train/train.py --data-root /data/images \\
        --teacher-cache /data/cache --out runs/v1 --resume runs/v1/last.pt

    # 3. smoke test on 200 images, no teacher download, 20 steps
    python tools/train/train.py --data-root /data/sample --teacher synthetic \\
        --out /tmp/smoke --max-steps 20 --batch-size 4 --num-workers 0
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

# Allow "python tools/train/train.py" from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from illumina.data import (  # noqa: E402
    DEFAULT_TEACHER_ID,
    AugmentConfig,
    DepthTeacher,
    DistillationDataset,
    TeacherCache,
    build_teacher,
    find_images,
    make_network_input,
    precompute_teacher_cache,
    split_paths,
)
from illumina.losses import align_prediction, midas_loss  # noqa: E402
from illumina.model import build_model  # noqa: E402

CHECKPOINT_FORMAT = "illumina-ckpt/1"
ARCH_NAME = "IlluminaDepth-448"

# Forward inference cost at 448x448, from `npx vite-node scripts/arch-report.ts`.
STUDENT_GFLOP_FORWARD = 2.198
# Backward is about twice the forward for a network of this shape.
STUDENT_GFLOP_TRAIN_STEP = STUDENT_GFLOP_FORWARD * 3.0
# Depth Anything V2 Small at 518x518: ViT-S/14 (1369 tokens, 12 blocks, d=384)
# plus a DPT head. Order-of-magnitude figure, not a measurement.
TEACHER_GFLOP_FORWARD = 90.0

# Rough sustained throughput in images/second at batch 32 in bf16. The first
# number assumes a precomputed teacher cache (student only), the second assumes
# the teacher runs online. These are estimates within about a factor of two and
# are replaced by the measured rate after the first epoch.
THROUGHPUT_TABLE: dict[str, tuple[float, float]] = {
    "H100 80GB": (2200.0, 330.0),
    "A100 40GB": (1300.0, 190.0),
    "RTX 4090": (1100.0, 160.0),
    "RTX 3090": (600.0, 90.0),
    "Apple M-series (mps)": (110.0, 14.0),
    "CPU (16 cores)": (6.0, 0.4),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Shows defaults and keeps the epilog's line breaks."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Distil a frozen monocular depth teacher into IlluminaDepth-448, a "
            "1.39 M parameter / 2.2 GFLOP student that runs in one WebGPU "
            "command encoder. Targets are relative inverse depth; the loss is "
            "scale- and shift-invariant, so nothing here is metric depth."
        ),
        epilog=__doc__.split("Examples\n--------\n", 1)[-1],
        formatter_class=_HelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        required=True,
        help="directory of source images, searched recursively",
    )
    data.add_argument(
        "--val-root",
        default=None,
        help="separate validation directory; overrides --val-fraction",
    )
    data.add_argument(
        "--val-fraction",
        type=float,
        default=0.02,
        help="fraction of --data-root held out for validation (hash split, "
        "stable when images are added)",
    )
    data.add_argument(
        "--input-size", type=int, default=448, help="must be a multiple of 32"
    )
    data.add_argument("--num-workers", type=int, default=8)
    data.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="use only the first N images; 0 means all (useful for smoke tests)",
    )

    aug = parser.add_argument_group("augmentation")
    aug.add_argument(
        "--scale-min", type=float, default=0.25, help="min crop area fraction"
    )
    aug.add_argument("--scale-max", type=float, default=1.0)
    aug.add_argument("--hflip-prob", type=float, default=0.5)
    aug.add_argument(
        "--jitter",
        type=float,
        default=0.35,
        help="brightness / contrast / saturation strength",
    )
    aug.add_argument("--hue", type=float, default=0.05, help="hue jitter, in turns")
    aug.add_argument(
        "--exposure-jitter",
        type=float,
        default=0.6,
        help="linear-light exposure jitter in stops; the deployed preprocess "
        "kernel takes the same knob, so the student must be robust to it",
    )

    teach = parser.add_argument_group("teacher")
    teach.add_argument(
        "--teacher",
        default="depth-anything-v2-small",
        help="depth-anything-v2-small | hf:<model-id> | synthetic "
        "(synthetic is a smoke-test stand-in and trains nothing useful)",
    )
    teach.add_argument(
        "--teacher-cache",
        default=None,
        help="directory of precomputed teacher disparity. Strongly recommended: "
        "the teacher costs ~40x a student training step, so an online teacher "
        "caps throughput at the teacher's speed for every epoch",
    )
    teach.add_argument(
        "--precompute",
        action="store_true",
        help="fill any missing --teacher-cache entries before training",
    )
    teach.add_argument(
        "--teacher-max-side", type=int, default=512, help="cached long side"
    )
    teach.add_argument(
        "--teacher-dtype", default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    teach.add_argument(
        "--teacher-batch-size", type=int, default=8, help="for --precompute"
    )

    optim = parser.add_argument_group("optimisation")
    optim.add_argument(
        "--batch-size", type=int, default=32, help="per optimiser micro-batch"
    )
    optim.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="micro-batches per optimiser step; effective batch is the product",
    )
    optim.add_argument("--epochs", type=int, default=30)
    optim.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after N optimiser steps; 0 means no cap",
    )
    optim.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    optim.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.02,
        help="cosine floor, as a fraction of --lr",
    )
    optim.add_argument("--weight-decay", type=float, default=0.02)
    optim.add_argument("--warmup-steps", type=int, default=1000)
    optim.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.99))
    optim.add_argument(
        "--grad-clip", type=float, default=1.0, help="0 disables clipping"
    )
    optim.add_argument(
        "--trim",
        type=float,
        default=0.2,
        help="fraction of the largest residuals dropped by the trimmed SSI loss",
    )
    optim.add_argument(
        "--grad-scales",
        type=int,
        default=4,
        help="scales K in the gradient-matching term (MiDaS uses 4)",
    )
    optim.add_argument(
        "--reg-weight",
        type=float,
        default=0.5,
        help="weight of the multi-scale gradient-matching term (MiDaS uses 0.5)",
    )
    optim.add_argument(
        "--supervise-at",
        default="head",
        choices=["head", "full"],
        help="'head' pools the target down to the head's resolution (224 for a "
        "448 input); 'full' bilinearly upsamples the prediction instead. 'head' "
        "matches deployment, where a joint bilateral filter -- not a bilinear "
        "upsample -- produces the full-resolution map",
    )
    optim.add_argument(
        "--ema-decay", type=float, default=0.9998, help="0 disables the EMA"
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    runtime.add_argument(
        "--amp",
        default="bf16",
        choices=["off", "fp16", "bf16"],
        help="mixed precision mode",
    )
    runtime.add_argument("--seed", type=int, default=1)
    runtime.add_argument(
        "--deterministic",
        action="store_true",
        help="reproducible augmentation and data order; costs some throughput",
    )
    runtime.add_argument(
        "--compile", action="store_true", help="wrap the student in torch.compile"
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--out", required=True, help="run directory for checkpoints and logs"
    )
    out.add_argument("--resume", default=None, help="checkpoint to resume from")
    out.add_argument(
        "--log-every", type=int, default=50, help="steps between log lines"
    )
    out.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="steps between validations; 0 means once per epoch",
    )
    out.add_argument(
        "--save-every", type=int, default=2000, help="steps between checkpoints"
    )
    out.add_argument(
        "--val-batches",
        type=int,
        default=0,
        help="cap validation batches; 0 means all",
    )
    out.add_argument(
        "--metric-min-disparity",
        type=float,
        default=0.02,
        help="disparity floor before inverting to depth for AbsRel/delta1; the "
        "metrics are only comparable across runs that share this value",
    )
    out.add_argument(
        "--dry-run", action="store_true", help="print the cost estimate and exit"
    )

    args = parser.parse_args(argv)
    if args.input_size % 32 != 0:
        parser.error(f"--input-size {args.input_size} must be a multiple of 32")
    if args.accum_steps < 1:
        parser.error("--accum-steps must be at least 1")
    if args.teacher_cache is None and args.precompute:
        parser.error("--precompute requires --teacher-cache")
    return args


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS needs this set before the first handle is created, or the
        # deterministic GEMM path raises at the first matmul instead of here.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def as_disparity(pred: Tensor) -> Tensor:
    """Normalises a model output to ``(N, 1, H, W)``.

    The deployed graph writes disparity into all four channels of a vec4
    texture, so accept ``(N, H, W)``, ``(N, 1, H, W)`` and ``(N, 4, H, W)``.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if pred.dim() != 4:
        raise ValueError(f"unexpected model output shape {tuple(pred.shape)}")
    return pred[:, :1] if pred.shape[1] > 1 else pred


def stem_in_channels(model: nn.Module) -> int:
    """Number of channels the stem convolution expects (4 in the shipped arch).

    Probed rather than assumed so this script keeps working whether ``model.py``
    consumes the 4-channel preprocessed tensor (normalised linear RGB plus the
    luminance guide, as ``reference.ts::refPreprocess`` produces) or a plain
    3-channel one.
    """
    for name, tensor in model.state_dict().items():
        if "stem" in name and tensor.dim() == 4:
            return int(tensor.shape[1])
    return 4


def unwrap(model: nn.Module) -> nn.Module:
    """Returns the underlying module behind a torch.compile wrapper."""
    return getattr(model, "_orig_mod", model)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class ModelEma:
    """Exponential moving average of the weights.

    ``theta_ema <- d * theta_ema + (1 - d) * theta`` over every floating-point
    entry of the state dict, buffers included, so BatchNorm running statistics
    track the averaged weights rather than the last minibatch. Integer entries
    (``num_batches_tracked``) are copied.

    The decay is warmed up as ``min(decay, (1 + step) / (10 + step))`` so the
    average is not dominated by the random initialisation for the first few
    thousand steps.

    An EMA is worth its memory here: with a cosine schedule and a batch of 32,
    the raw weights at any given step are noticeably noisier than their average,
    and the exported model is a snapshot, not an ensemble.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(unwrap(model)).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        source = unwrap(model).state_dict()
        for key, value in self.module.state_dict().items():
            incoming = source[key]
            if value.is_floating_point():
                value.mul_(decay).add_(
                    incoming.detach().to(value.dtype), alpha=1 - decay
                )
            else:
                value.copy_(incoming)

    def state_dict(self) -> dict[str, Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.module.load_state_dict(state)


# ---------------------------------------------------------------------------
# Optimiser and schedule
# ---------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> torch.optim.Optimizer:
    """AdamW with decay on weight matrices only.

    Decaying biases and BatchNorm scales pulls the normalisation towards a
    degenerate identity and measurably hurts small networks, so every parameter
    with fewer than two dimensions goes in a no-decay group.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), eps=1e-8)


def cosine_warmup(
    step: int, warmup_steps: int, total_steps: int, min_ratio: float
) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base LR."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------


@dataclass
class LossTerms:
    total: Tensor
    ssi: Tensor
    reg: Tensor


def prepare_pair(
    pred: Tensor,
    target: Tensor,
    supervise_at: str,
) -> tuple[Tensor, Tensor]:
    """Brings prediction and target to a common resolution, as ``(N, H, W)``.

    ``head`` pools the target down to the prediction's resolution with an
    antialiased bilinear filter. That is the honest choice for this network:
    the head emits a half-resolution map and the deployed pipeline recovers full
    resolution with an edge-aware joint bilateral filter, so supervising a
    bilinear upsample would train the student to compensate for an upsampler
    that is not the one it ships with.
    """
    pred = as_disparity(pred)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if pred.shape[-2:] != target.shape[-2:]:
        if supervise_at == "head":
            target = F.interpolate(
                target,
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        else:
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
    return pred.squeeze(1), target.squeeze(1)


def distillation_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    trim: float,
    reg_weight: float,
    scales: int,
) -> LossTerms:
    """``L = L_ssi(trimmed) + reg_weight * L_reg``, computed in float32.

    Delegates to ``illumina.losses.midas_loss``, which solves the least-squares
    affine alignment once and feeds the *aligned* prediction to both terms.
    That sharing is not just an optimisation: the gradient-matching term is only
    scale-invariant if it sees an aligned prediction, so computing it on the raw
    output would quietly penalise the prediction's arbitrary scale.

    Called outside autocast -- the alignment accumulates sums over ~50 k pixels
    per image, where fp16 loses too much precision.
    """
    total, parts = midas_loss(
        pred.float(),
        target.float(),
        mask,
        alpha=reg_weight,
        scales=scales,
        trim=trim,
    )
    return LossTerms(total=total, ssi=parts["ssi"], reg=parts["reg"])


def build_mask(target: Tensor, min_valid: float = 0.0) -> Tensor:
    """Boolean validity mask, ``(N, H, W)``.

    Teacher disparity is dense, so this only rejects non-finite values (which a
    half-precision teacher can produce on degenerate inputs) and, optionally,
    everything at or below ``min_valid`` -- the flat floor that teachers assign
    to sky.
    """
    mask = torch.isfinite(target)
    if min_valid > 0.0:
        mask = mask & (target > min_valid)
    return mask


def depth_metrics(
    aligned_disp: Tensor,
    target_disp: Tensor,
    mask: Tensor,
    min_disparity: float,
) -> tuple[float, float, float]:
    """AbsRel, delta1 and the mean absolute disparity error.

    Disparity is inverted to depth after clamping to ``min_disparity``: the
    teacher's far field sits near zero disparity, where 1/d explodes and the
    ratio metrics stop meaning anything. The clamp is therefore part of the
    metric definition, and the numbers are only comparable between runs that
    share it. These are teacher-agreement metrics, not ground-truth accuracy.

        AbsRel = mean(|d_pred - d_gt| / d_gt)
        delta1 = fraction of pixels with max(d_pred/d_gt, d_gt/d_pred) < 1.25
    """
    valid = mask & torch.isfinite(aligned_disp)
    count = int(valid.sum())
    if count == 0:
        return float("nan"), float("nan"), float("nan")

    pred_d = 1.0 / aligned_disp.clamp_min(min_disparity)
    gt_d = 1.0 / target_disp.clamp_min(min_disparity)

    abs_rel = ((pred_d - gt_d).abs() / gt_d)[valid].mean()
    ratio = torch.maximum(pred_d / gt_d, gt_d / pred_d)[valid]
    delta1 = (ratio < 1.25).float().mean()
    disp_mae = (aligned_disp - target_disp).abs()[valid].mean()
    return float(abs_rel), float(delta1), float(disp_mae)


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------


def make_input(
    batch: dict[str, Tensor],
    device: torch.device,
    in_channels: int,
) -> Tensor:
    """Builds the network input tensor from a dataloader batch.

    Mirrors ``reference.ts::refPreprocess`` exactly: sRGB decoded with the
    piecewise EOTF, multiplied by the per-sample exposure, normalised with the
    ImageNet statistics *in linear light*, with luminance in channel 3.
    """
    rgb = batch["student_rgb"].to(device, non_blocking=True)
    exposure = batch["exposure"].to(device, non_blocking=True)
    full = make_network_input(rgb, exposure=exposure)
    return full if in_channels == 4 else full[:, :in_channels]


def resolve_target(
    batch: dict[str, Tensor],
    device: torch.device,
    teacher: DepthTeacher | None,
) -> Tensor:
    """Returns ``(N, 1, S, S)`` teacher disparity, from cache or run online.

    The dataset signals "no cached target" with a zero-channel tensor, which
    the default collate stacks without complaint.
    """
    target = batch["target"]
    if target.shape[1] == 1:
        return target.to(device, non_blocking=True)
    if teacher is None:
        raise RuntimeError(
            "no cached targets and no online teacher; pass --teacher-cache "
            "(with --precompute) or a working --teacher"
        )
    clean = batch["teacher_rgb"].to(device, non_blocking=True)
    disparity = teacher(clean)
    # The teacher runs under torch.inference_mode, and an inference tensor
    # cannot take part in an autograd graph ("Inference tensors cannot be saved
    # for backward"). Cloning it *outside* that context is the documented way
    # back to an ordinary tensor, and it must happen here rather than inside
    # the teacher, where the clone would still be an inference tensor.
    return disparity.clone().detach()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    teacher: DepthTeacher | None,
    in_channels: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    """Runs validation and returns averaged metrics."""
    was_training = model.training
    model.eval()

    totals = {"ssi": 0.0, "reg": 0.0, "abs_rel": 0.0, "delta1": 0.0, "disp_mae": 0.0}
    seen = 0
    for index, batch in enumerate(loader):
        if args.val_batches and index >= args.val_batches:
            break
        inputs = make_input(batch, device, in_channels)
        target = resolve_target(batch, device, teacher)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            pred = model(inputs)
        pred, target_r = prepare_pair(pred.float(), target, args.supervise_at)
        mask = build_mask(target_r)
        terms = distillation_loss(
            pred, target_r, mask, args.trim, args.reg_weight, args.grad_scales
        )

        # Same closed form the objective uses, so the reported numbers and
        # the optimised loss agree about what "aligned" means.
        aligned = align_prediction(pred.float(), target_r, mask)
        abs_rel, delta1, disp_mae = depth_metrics(
            aligned, target_r, mask, args.metric_min_disparity
        )

        count = pred.shape[0]
        seen += count
        totals["ssi"] += float(terms.ssi) * count
        totals["reg"] += float(terms.reg) * count
        for key, value in (
            ("abs_rel", abs_rel),
            ("delta1", delta1),
            ("disp_mae", disp_mae),
        ):
            if not math.isnan(value):
                totals[key] += value * count

    if was_training:
        model.train()
    if seen == 0:
        return {key: float("nan") for key in totals}
    metrics = {key: value / seen for key, value in totals.items()}
    metrics["loss"] = metrics["ssi"] + args.reg_weight * metrics["reg"]
    return metrics


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEma | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    teacher_name: str,
    best_metric: float = float("inf"),
) -> None:
    """Writes a checkpoint atomically (temp file plus rename).

    A checkpoint half-written by a job that hit its wall-clock limit is worse
    than no checkpoint, because it fails to load only at the point where the
    next job has already queued and waited.
    """
    payload = {
        "format": CHECKPOINT_FORMAT,
        "arch": ARCH_NAME,
        "input_size": args.input_size,
        "with_bn": True,
        "step": step,
        "epoch": epoch,
        "model": unwrap(model).state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "metrics": metrics,
        "best_metric": best_metric,
        "args": vars(args),
        "teacher": teacher_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema: ModelEma | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, int, float]:
    """Restores a run. Returns ``(step, epoch, best_metric)``.

    The best metric travels with the checkpoint because a resumed run that
    started from ``inf`` would overwrite a good ``best.pt`` with the first
    validation it happens to do, however bad.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} checkpoint "
            f"(found {state.get('format')!r})"
        )
    unwrap(model).load_state_dict(state["model"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return (
        int(state.get("step", 0)),
        int(state.get("epoch", 0)),
        float(state.get("best_metric", float("inf"))),
    )


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------


def print_cost_estimate(
    args: argparse.Namespace,
    n_train: int,
    n_val: int,
    cached: bool,
    missing_cache: int,
) -> None:
    """Prints an honest, clearly-labelled estimate of what this run will cost."""
    scale = (args.input_size / 448.0) ** 2
    step_gflop = STUDENT_GFLOP_TRAIN_STEP * scale
    samples_per_epoch = (n_train // args.batch_size) * args.batch_size
    total_samples = samples_per_epoch * args.epochs
    if args.max_steps:
        capped = args.max_steps * args.batch_size * args.accum_steps
        total_samples = min(total_samples, capped)

    print("=" * 72)
    print("Expected cost -- estimates, accurate to about a factor of two.")
    print("The measured rate after the first epoch supersedes all of this.")
    print("-" * 72)
    print(f"  train images            : {n_train:,}")
    print(f"  val images              : {n_val:,}")
    print(f"  samples per epoch       : {samples_per_epoch:,}")
    print(f"  epochs                  : {args.epochs}")
    print(f"  total samples seen      : {total_samples:,}")
    print(f"  effective batch         : {args.batch_size * args.accum_steps}")
    print(
        f"  student fwd (inference) : "
        f"{STUDENT_GFLOP_FORWARD * scale:.2f} GFLOP/image"
    )
    print(f"  student train step      : {step_gflop:.2f} GFLOP/image (fwd + bwd)")
    print(f"  total student compute   : {total_samples * step_gflop / 1e6:.1f} PFLOP")
    if cached:
        print(f"  teacher                 : cached ({missing_cache:,} entries missing)")
        if missing_cache:
            cache_pflop = missing_cache * TEACHER_GFLOP_FORWARD / 1e6
            print(f"  cache still to build    : {cache_pflop:.2f} PFLOP one-off")
    else:
        print(
            f"  teacher                 : ONLINE, ~{TEACHER_GFLOP_FORWARD:.0f} "
            "GFLOP/image every epoch"
        )
        print(
            f"  total teacher compute   : "
            f"{total_samples * TEACHER_GFLOP_FORWARD / 1e6:.1f} PFLOP "
            f"({TEACHER_GFLOP_FORWARD / step_gflop:.0f}x the student)"
        )
    print("-" * 72)
    print("  Wall clock, assuming the input pipeline keeps up:")
    column = 0 if cached else 1
    for name, rates in THROUGHPUT_TABLE.items():
        rate = rates[column]
        hours = total_samples / rate / 3600.0
        print(f"    {name:<22}{rate:>7.0f} img/s   {hours:>8.1f} h")
    print("-" * 72)
    print("  Reality checks:")
    print("    - With a cache, this run is almost certainly bound by JPEG decode,")
    print("      not by the GPU. One dataloader worker decodes and augments about")
    print("      150-250 images/s, so plan on --num-workers 8 or more.")
    print("    - A usable model needs roughly 1-3 M samples seen over at least")
    print("      ~50 k distinct source images. Below that the student memorises")
    print("      the teacher's quirks on the training set.")
    print("    - Targets are relative disparity from a teacher. AbsRel and delta1")
    print("      below measure agreement with the teacher, not metric accuracy.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def collect_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Resolves the train and validation image lists.

    Done once, before anything expensive, so the cost estimate quotes the real
    counts and a large image root is walked a single time.
    """
    paths = find_images(args.data_root)
    if args.limit_images:
        paths = paths[: args.limit_images]

    if args.val_root:
        val_paths = find_images(args.val_root)
        if args.limit_images:
            val_paths = val_paths[: max(1, args.limit_images // 10)]
        return paths, val_paths
    return split_paths(paths, args.val_fraction, args.seed)


def make_loaders(
    args: argparse.Namespace,
    cache: TeacherCache | None,
    train_paths: Sequence[Path],
    val_paths: Sequence[Path],
) -> tuple[DataLoader, DataLoader | None]:
    """Builds the train and validation loaders."""
    train_aug = AugmentConfig(
        scale=(args.scale_min, args.scale_max),
        hflip_prob=args.hflip_prob,
        brightness=args.jitter,
        contrast=args.jitter,
        saturation=args.jitter,
        hue=args.hue,
        exposure_log2=args.exposure_jitter,
    )
    train_set = DistillationDataset(
        train_paths,
        size=args.input_size,
        augment=train_aug,
        cache=cache,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    val_set = (
        DistillationDataset(
            val_paths,
            size=args.input_size,
            augment=AugmentConfig.deterministic(),
            cache=cache,
            deterministic=True,
            seed=args.seed,
        )
        if val_paths
        else None
    )

    pin = args.device.startswith("cuda")
    common = {
        "num_workers": args.num_workers,
        "pin_memory": pin,
        # set_epoch cannot reach forked workers that outlive an epoch, so a
        # deterministic run keeps workers ephemeral.
        "persistent_workers": args.num_workers > 0 and not args.deterministic,
    }
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        )
        if val_set is not None
        else None
    )
    return train_loader, val_loader


def append_log(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = None if args.amp == "off" else dtypes[args.amp]
    if amp_dtype is not None and device.type == "cpu":
        print("cpu device: disabling mixed precision")
        amp_dtype = None

    # --- teacher and cache -------------------------------------------------
    teacher = None
    cache: TeacherCache | None = None
    teacher_name = args.teacher
    teacher_dtype = dtypes[args.teacher_dtype]
    if device.type == "cpu" and teacher_dtype is not torch.float32:
        print("cpu device: forcing the teacher to fp32")
        teacher_dtype = torch.float32

    if args.teacher_cache:
        # The cache key includes the teacher name, so it has to be resolved
        # before anything can be looked up in the cache.
        resolved = (
            "synthetic"
            if args.teacher == "synthetic"
            else (args.teacher[3:] if args.teacher.startswith("hf:") else None)
        )
        if resolved is None:
            resolved = DEFAULT_TEACHER_ID
        teacher_name = resolved
        cache = TeacherCache(args.teacher_cache, resolved, args.teacher_max_side)

    train_paths, val_paths = collect_paths(args)
    all_paths = list(train_paths) + list(val_paths)
    missing = 0
    if cache is not None:
        missing = sum(1 for p in all_paths if not cache.has(p))

    print_cost_estimate(
        args,
        n_train=len(train_paths),
        n_val=len(val_paths),
        cached=cache is not None,
        missing_cache=missing,
    )
    if args.dry_run:
        return 0

    if cache is not None and missing:
        if not args.precompute:
            raise SystemExit(
                f"{missing} of {len(all_paths)} images have no cache entry in "
                f"{args.teacher_cache}. Re-run with --precompute, or build the "
                "cache first:\n"
                f"    python -m illumina.data --root {args.data_root} "
                f"--cache-dir {args.teacher_cache}"
            )
        builder = build_teacher(args.teacher, device=device, dtype=teacher_dtype)
        precompute_teacher_cache(
            all_paths,
            builder,
            cache,
            batch_size=args.teacher_batch_size,
            device=device,
        )
        del builder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif cache is None:
        teacher = build_teacher(args.teacher, device=device, dtype=teacher_dtype)
        teacher_name = teacher.name

    # --- data --------------------------------------------------------------
    train_loader, val_loader = make_loaders(args, cache, train_paths, val_paths)
    print(f"data: {len(train_paths):,} train / {len(val_paths):,} val images")

    # --- model -------------------------------------------------------------
    model = build_model(input_size=args.input_size, with_bn=True).to(device)
    in_channels = stem_in_channels(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model: {n_params:,} trainable parameters, "
        f"stem expects {in_channels} channels"
    )
    if args.compile:
        model = torch.compile(model)

    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None
    optimizer = build_optimizer(model, args.lr, args.weight_decay, tuple(args.betas))

    steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup(
            step, args.warmup_steps, total_steps, args.min_lr_ratio
        ),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)

    step = 0
    start_epoch = 0
    best_metric = float("inf")
    if args.resume:
        step, start_epoch, best_metric = load_checkpoint(
            args.resume, model, ema, optimizer, scheduler, scaler, device
        )
        print(
            f"resumed from {args.resume} at step {step}, epoch {start_epoch}"
            + (f", best AbsRel {best_metric:.4f}" if best_metric < float("inf") else "")
        )

    window_samples = 0
    window_start = time.time()
    stop = False
    completed_epoch = start_epoch

    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        train_loader.dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        epoch_samples = 0

        for micro, batch in enumerate(train_loader):
            inputs = make_input(batch, device, in_channels)
            target = resolve_target(batch, device, teacher)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                pred = model(inputs)
            # The loss runs outside autocast: the scale/shift solve sums over
            # ~50 k pixels and loses too much precision in fp16.
            pred_r, target_r = prepare_pair(pred.float(), target, args.supervise_at)
            mask = build_mask(target_r)
            terms = distillation_loss(
                pred_r, target_r, mask, args.trim, args.reg_weight, args.grad_scales
            )
            scaler.scale(terms.total / args.accum_steps).backward()

            epoch_samples += inputs.shape[0]
            window_samples += inputs.shape[0]

            if (micro + 1) % args.accum_steps != 0:
                continue

            grad_norm = float("nan")
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)
            step += 1

            if args.log_every and step % args.log_every == 0:
                elapsed = max(time.time() - window_start, 1e-6)
                rate = window_samples / elapsed
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(terms.total),
                    "ssi": float(terms.ssi),
                    "reg": float(terms.reg),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "img_per_s": rate,
                }
                print(
                    f"step {step:>7} epoch {epoch:>3} loss {record['loss']:.4f} "
                    f"(ssi {record['ssi']:.4f} reg {record['reg']:.4f}) "
                    f"lr {record['lr']:.2e} grad {grad_norm:.2f} "
                    f"{rate:.0f} img/s"
                )
                append_log(log_path, record)
                window_samples = 0
                window_start = time.time()

            due = args.val_every and step % args.val_every == 0
            if due and val_loader is not None:
                best_metric = _validate_and_save(
                    model, ema, val_loader, device, args, teacher, in_channels,
                    amp_dtype, optimizer, scheduler, scaler, step, epoch,
                    out_dir, log_path, best_metric, teacher_name,
                )

            if args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    out_dir / "last.pt", model, ema, optimizer, scheduler, scaler,
                    step, epoch, {}, args, teacher_name, best_metric,
                )

            if args.max_steps and step >= args.max_steps:
                print(f"reached --max-steps {args.max_steps}")
                stop = True
                break

        # An epoch cut short by --max-steps is not a completed epoch: recording
        # it as one would make a resume skip the rest of its data.
        completed_epoch = epoch if stop else epoch + 1
        epoch_time = time.time() - epoch_start
        rate = epoch_samples / max(epoch_time, 1e-6)
        remaining = max(args.epochs - epoch - 1, 0)
        print(
            f"epoch {epoch} done in {epoch_time / 60:.1f} min "
            f"({rate:.0f} img/s measured); "
            f"~{remaining * epoch_time / 3600:.1f} h left at this rate"
        )
        # A trailing partial accumulation window would otherwise leak gradients
        # from this epoch into the first step of the next one.
        optimizer.zero_grad(set_to_none=True)

        if val_loader is not None:
            best_metric = _validate_and_save(
                model, ema, val_loader, device, args, teacher, in_channels,
                amp_dtype, optimizer, scheduler, scaler, step, completed_epoch,
                out_dir, log_path, best_metric, teacher_name,
            )
        save_checkpoint(
            out_dir / "last.pt", model, ema, optimizer, scheduler, scaler,
            step, completed_epoch, {}, args, teacher_name, best_metric,
        )

    save_checkpoint(
        out_dir / "last.pt", model, ema, optimizer, scheduler, scaler,
        step, completed_epoch, {}, args, teacher_name, best_metric,
    )
    best_path = out_dir / "best.pt"
    export_from = best_path if best_path.is_file() else out_dir / "last.pt"
    print(f"done. checkpoints in {out_dir}")
    if not best_path.is_file():
        print("no best.pt: nothing was validated (no validation split)")
    print("Export with:")
    print(
        f"    python tools/export/export_idm.py --checkpoint {export_from} "
        "--out public/weights.idm"
    )
    return 0


def _validate_and_save(
    model: nn.Module,
    ema: ModelEma | None,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    teacher: DepthTeacher | None,
    in_channels: int,
    amp_dtype: torch.dtype | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    epoch: int,
    out_dir: Path,
    log_path: Path,
    best_metric: float,
    teacher_name: str,
) -> float:
    """Validates the raw and EMA weights, logs both, keeps the better one.

    Model selection uses the EMA's AbsRel when an EMA exists, because that is
    what gets exported.
    """
    raw = evaluate(model, val_loader, device, args, teacher, in_channels, amp_dtype)
    line = (
        f"  val   step {step:>7} raw: loss {raw['loss']:.4f} "
        f"AbsRel {raw['abs_rel']:.4f} d1 {raw['delta1']:.4f}"
    )
    record: dict[str, Any] = {"step": step, "epoch": epoch, "val_raw": raw}

    selection = raw["abs_rel"]
    if ema is not None:
        ema_metrics = evaluate(
            ema.module, val_loader, device, args, teacher, in_channels, amp_dtype
        )
        line += (
            f" | ema: loss {ema_metrics['loss']:.4f} "
            f"AbsRel {ema_metrics['abs_rel']:.4f} d1 {ema_metrics['delta1']:.4f}"
        )
        record["val_ema"] = ema_metrics
        selection = ema_metrics["abs_rel"]

    print(line)
    append_log(log_path, record)

    if not math.isnan(selection) and selection < best_metric:
        best_metric = selection
        save_checkpoint(
            out_dir / "best.pt", model, ema, optimizer, scheduler, scaler,
            step, epoch, record, args, teacher_name, best_metric,
        )
        print(f"  new best AbsRel {best_metric:.4f} -> {out_dir / 'best.pt'}")
    return best_metric


if __name__ == "__main__":
    sys.exit(main())
