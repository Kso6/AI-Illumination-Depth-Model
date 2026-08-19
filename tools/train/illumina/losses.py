"""MiDaS scale- and shift-invariant losses for disparity distillation.

IlluminaDepth predicts *relative* inverse depth (disparity). A relative
prediction is only defined up to an affine transform: a network that outputs
``2 * d + 0.3`` has understood the scene exactly as well as one that outputs
``d``. Training it with a plain L1 or L2 loss would therefore spend most of its
capacity chasing a global scale and offset that nobody asked for, and would
make it impossible to mix teachers or datasets whose disparity ranges differ.

The fix, from MiDaS, is to align the prediction to the target with the
least-squares optimal scale and shift *inside the loss*, and only then measure
the residual:

    Ranftl, Lasinger, Hafner, Schindler and Koltun,
    "Towards Robust Monocular Depth Estimation: Mixing Datasets for
     Zero-shot Cross-dataset Transfer",
    IEEE TPAMI 44(3):1623-1637, 2022 (arXiv:1907.01341).

Two terms are combined, exactly as in the paper:

* ``L_ssi``  -- a trimmed absolute deviation on the aligned prediction. The
  largest 20 % of residuals are dropped, so a handful of pixels where the
  teacher is confidently wrong (sky, mirrors, thin structures) cannot dominate
  the gradient.
* ``L_reg``  -- a multi-scale gradient-matching term over K = 4 scales, which
  is what actually makes depth edges sharp: it penalises differences between
  the *derivatives* of prediction and target, so the student is pushed to put
  its discontinuities exactly where the teacher put its own.

    L = L_ssi + 0.5 * L_reg

Conventions used throughout this module:

* tensors are ``[N, H, W]`` or ``[N, 1, H, W]`` (the channel axis is squeezed);
* ``mask`` is 1 where the target is valid and 0 elsewhere; if it is omitted,
  every pixel is treated as valid;
* the default reduction is MiDaS's "batch based": the per-image sums are added
  and divided by the *total* number of valid pixels, so images with more valid
  pixels count for more. ``reduction_image_based`` normalises each image first
  and then averages, which is the right choice when validity varies wildly
  between samples.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn

__all__ = [
    "Reduction",
    "reduction_batch_based",
    "reduction_image_based",
    "compute_scale_and_shift",
    "align_prediction",
    "masked_mse_loss",
    "trimmed_mae_loss",
    "ssi_loss",
    "gradient_loss",
    "gradient_matching_loss",
    "midas_loss",
    "MidasLoss",
]

Reduction = Callable[[Tensor, Tensor], Tensor]


# ---------------------------------------------------------------------------
# Shapes and reductions
# ---------------------------------------------------------------------------


def _as_bhw(x: Tensor, name: str) -> Tensor:
    """Normalises ``[N, 1, H, W]`` or ``[N, H, W]`` down to ``[N, H, W]``."""
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError(f"{name}: expected 1 channel, got {x.shape[1]}")
        return x[:, 0]
    if x.dim() == 3:
        return x
    raise ValueError(f"{name}: expected [N, H, W] or [N, 1, H, W], got {tuple(x.shape)}")


def _prepare(
    prediction: Tensor, target: Tensor, mask: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    """Squeezes all three tensors to ``[N, H, W]`` and materialises the mask."""
    pred = _as_bhw(prediction, "prediction")
    tgt = _as_bhw(target, "target")
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction {tuple(pred.shape)} != target {tuple(tgt.shape)}")
    if mask is None:
        m = torch.ones_like(tgt)
    else:
        m = _as_bhw(mask, "mask").to(dtype=tgt.dtype)
        if m.shape != tgt.shape:
            raise ValueError(f"mask {tuple(m.shape)} != target {tuple(tgt.shape)}")
    return pred, tgt, m


def reduction_batch_based(image_loss: Tensor, valid: Tensor) -> Tensor:
    """Sum of per-image losses divided by the total number of valid pixels."""
    divisor = torch.sum(valid)
    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss: Tensor, valid: Tensor) -> Tensor:
    """Mean over images of (per-image loss / per-image valid count)."""
    valid = valid.clone()
    empty = valid == 0
    valid[empty] = 1.0
    per_image = image_loss / valid
    per_image = torch.where(empty, torch.zeros_like(per_image), per_image)
    return torch.mean(per_image)


# ---------------------------------------------------------------------------
# Least-squares alignment
# ---------------------------------------------------------------------------


def compute_scale_and_shift(
    prediction: Tensor, target: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Per-image least-squares scale ``s`` and shift ``t``.

    Solves, independently for each image in the batch,

        (s, t) = argmin  sum_i  m_i * (s * d_i + t - d*_i) ^ 2

    which is a 2 x 2 linear system --- the normal equations of the design
    matrix ``[d, 1]``:

        | sum m d^2   sum m d | | s |   | sum m d d* |
        |                     | |   | = |            |
        | sum m d     sum m   | | t |   | sum m d*   |

    written ``A x = b`` with ``A = [[a00, a01], [a01, a11]]``. Cramer's rule
    gives the closed form used below,

        det = a00 * a11 - a01 * a01
        s   = ( a11 * b0 - a01 * b1) / det
        t   = (-a01 * b0 + a00 * b1) / det

    which is far cheaper and better conditioned than calling a generic solver
    once per image. ``det`` is zero exactly when the prediction is constant over
    the valid region (or the region is empty); those images get ``s = t = 0``,
    contributing nothing to the loss rather than an infinity.

    Gradients flow through ``s`` and ``t``, as in the reference implementation:
    the alignment is part of the model's objective, not a detached
    postprocessing step.

    Returns ``(scale, shift)``, each of shape ``[N]``.
    """
    pred, tgt, m = _prepare(prediction, target, mask)

    a_00 = torch.sum(m * pred * pred, (1, 2))
    a_01 = torch.sum(m * pred, (1, 2))
    a_11 = torch.sum(m, (1, 2))

    b_0 = torch.sum(m * pred * tgt, (1, 2))
    b_1 = torch.sum(m * tgt, (1, 2))

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    # Guard the division itself, not just the result: dividing by a zero
    # determinant would put NaNs into the backward pass even where the forward
    # value is later masked away.
    safe_det = torch.where(valid, det, torch.ones_like(det))
    scale = torch.where(valid, (a_11 * b_0 - a_01 * b_1) / safe_det, torch.zeros_like(det))
    shift = torch.where(valid, (-a_01 * b_0 + a_00 * b_1) / safe_det, torch.zeros_like(det))
    return scale, shift


def align_prediction(
    prediction: Tensor, target: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Returns ``s * prediction + t`` with the least-squares optimal ``s``, ``t``."""
    pred, _, _ = _prepare(prediction, target, mask)
    scale, shift = compute_scale_and_shift(prediction, target, mask)
    return scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)


# ---------------------------------------------------------------------------
# Data terms
# ---------------------------------------------------------------------------


def masked_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    reduction: Reduction = reduction_batch_based,
) -> Tensor:
    """The untrimmed data term, ``(1 / 2M) * sum_i m_i (d_i - d*_i) ^ 2``.

    The factor of two in the denominator is the paper's; it is a constant and
    matters only in that ``alpha = 0.5`` for the regulariser was tuned with it.
    """
    pred, tgt, m = _prepare(prediction, target, mask)
    valid = torch.sum(m, (1, 2))
    res = pred - tgt
    image_loss = torch.sum(m * res * res, (1, 2))
    return reduction(image_loss, 2.0 * valid)


def trimmed_mae_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    trim: float = 0.2,
) -> Tensor:
    """Trimmed absolute deviation, the paper's robust data term.

        L_ssitrim = (1 / 2M) * sum_{j = 1}^{U} |r_j|,
        r sorted ascending,  U = (1 - trim) * M,  M = number of valid pixels

    Note that the normaliser keeps the *full* count ``M`` even though only
    ``U`` residuals are summed: dropping the worst 20 % is meant to remove their
    gradient, not to inflate the loss of everything that is left.

    Trimming is done over the whole batch at once, matching the reference
    implementation, so one badly-labelled image can have more of its pixels
    trimmed than a clean one. That is the intended behaviour --- it is a per-
    dataset outlier rejector, not a per-image one.
    """
    if not 0.0 <= trim < 1.0:
        raise ValueError(f"trim must be in [0, 1), got {trim}")
    pred, tgt, m = _prepare(prediction, target, mask)

    total_valid = torch.sum(m)
    if total_valid == 0:
        return torch.sum(pred) * 0.0

    res = (pred - tgt)[m > 0].abs()
    keep = int(res.numel() * (1.0 - trim))
    if keep <= 0:
        return torch.sum(pred) * 0.0
    trimmed, _ = torch.sort(res)
    return torch.sum(trimmed[:keep]) / (2.0 * total_valid)


def ssi_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    trim: float = 0.2,
    align: bool = True,
    reduction: Reduction = reduction_batch_based,
) -> Tensor:
    """Scale- and shift-invariant data term.

    With ``align=True`` (the default) the prediction is first mapped onto the
    target by the closed-form least-squares affine fit of
    :func:`compute_scale_and_shift`, which is what makes the loss invariant to
    the prediction's arbitrary scale and offset. Pass ``align=False`` when the
    caller has already aligned --- :func:`midas_loss` does, so the alignment is
    solved once and shared with the gradient term.

    ``trim > 0`` selects the robust trimmed-MAE variant of the paper;
    ``trim = 0`` falls back to the plain masked MSE. ``reduction`` applies only
    to the MSE variant --- the trimmed variant is batch-based by construction
    (see :func:`trimmed_mae_loss`).

    A note on faithfulness: the paper pairs its *trimmed* loss with a robust
    median / mean-absolute-deviation normalisation of both prediction and
    target, and uses the least-squares fit for the MSE variant. Here the
    least-squares fit is used in both cases, because the teacher's disparity is
    dense and well behaved and because sharing one alignment with the gradient
    term keeps the two loss components consistent with each other.
    """
    pred = align_prediction(prediction, target, mask) if align else prediction
    if trim > 0.0:
        return trimmed_mae_loss(pred, target, mask, trim=trim)
    return masked_mse_loss(pred, target, mask, reduction=reduction)


# ---------------------------------------------------------------------------
# Gradient matching
# ---------------------------------------------------------------------------


def gradient_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    reduction: Reduction = reduction_batch_based,
) -> Tensor:
    """First-order gradient-matching term at one scale.

        L = (1 / M) * sum_i ( |dx R_i| + |dy R_i| ),   R = d - d*

    Derivatives are one-sided finite differences. A difference is only counted
    where *both* of the pixels it spans are valid, otherwise every boundary of
    the valid region would register as a huge fake edge.
    """
    pred, tgt, m = _prepare(prediction, target, mask)
    valid = torch.sum(m, (1, 2))

    diff = m * (pred - tgt)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = m[:, :, 1:] * m[:, :, :-1]
    grad_x = mask_x * grad_x

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = m[:, 1:, :] * m[:, :-1, :]
    grad_y = mask_y * grad_y

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    return reduction(image_loss, valid)


def gradient_matching_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    scales: int = 4,
    align: bool = False,
    reduction: Reduction = reduction_batch_based,
) -> Tensor:
    """Multi-scale gradient matching, summed over K scales.

        L_reg = sum_{k = 1}^{K} L_grad( d^k, d*^k )

    where level ``k`` is obtained by subsampling with stride ``2^(k-1)``, so
    ``k = 1`` is full resolution and ``K = 4`` reaches a stride of 8. Plain
    strided subsampling is used rather than area averaging, exactly as in the
    reference implementation: pooling would smooth away the very discontinuities
    this term exists to sharpen.

    ``align=True`` applies the least-squares affine fit first; the default is
    ``False`` because :func:`midas_loss` aligns once and passes the result in.
    """
    if scales < 1:
        raise ValueError(f"scales must be >= 1, got {scales}")
    pred, tgt, m = _prepare(
        align_prediction(prediction, target, mask) if align else prediction, target, mask
    )

    total = torch.zeros((), dtype=pred.dtype, device=pred.device)
    for k in range(scales):
        step = 2**k
        total = total + gradient_loss(
            pred[:, ::step, ::step],
            tgt[:, ::step, ::step],
            m[:, ::step, ::step],
            reduction=reduction,
        )
    return total


# ---------------------------------------------------------------------------
# The combined objective
# ---------------------------------------------------------------------------


def midas_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    alpha: float = 0.5,
    scales: int = 4,
    trim: float = 0.2,
    reduction: Reduction = reduction_batch_based,
) -> tuple[Tensor, dict[str, Tensor]]:
    """``L = L_ssi + alpha * L_reg``, the full MiDaS training objective.

    The affine alignment is solved once and used for both terms, so the data
    term and the gradient term always measure the same aligned prediction.

    Returns ``(total, parts)`` where ``parts`` holds the detached ``ssi`` and
    ``reg`` components for logging.
    """
    aligned = align_prediction(prediction, target, mask)
    data = ssi_loss(aligned, target, mask, trim=trim, align=False, reduction=reduction)
    reg = gradient_matching_loss(
        aligned, target, mask, scales=scales, align=False, reduction=reduction
    )
    total = data + alpha * reg
    return total, {"ssi": data.detach(), "reg": reg.detach()}


class MidasLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`midas_loss`.

    ``alpha = 0.5`` and ``K = 4`` scales are the paper's values and are what the
    training script defaults to. ``trim = 0.2`` drops the worst fifth of the
    residuals; set ``trim = 0`` for the plain squared-error variant, which
    converges slightly faster on a clean synthetic teacher but is noticeably
    worse on real photographs.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        scales: int = 4,
        trim: float = 0.2,
        reduction: Reduction = reduction_batch_based,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.scales = scales
        self.trim = trim
        self.reduction = reduction

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        return_parts: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        total, parts = midas_loss(
            prediction,
            target,
            mask,
            alpha=self.alpha,
            scales=self.scales,
            trim=self.trim,
            reduction=self.reduction,
        )
        if return_parts:
            return total, parts
        return total

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, scales={self.scales}, trim={self.trim}"
