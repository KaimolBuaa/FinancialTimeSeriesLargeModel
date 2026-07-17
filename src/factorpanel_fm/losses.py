"""Masked and cross-sectional objectives for FactorPanel-FM training."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import torch
from torch.nn import functional as F


def _validate_matching_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (prediction, target, mask)):
        raise TypeError("prediction, target, and mask must be torch.Tensor instances")
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, target, and mask must have matching shapes")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must have floating-point dtypes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have bool dtype")
    if prediction.device != target.device or mask.device != prediction.device:
        raise ValueError("prediction, target, and mask must be on the same device")


def _graph_zero(tensor: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(tensor), tensor, 0.0).sum() * 0.0


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Return mean Huber loss over masked finite elements."""

    _validate_matching_tensors(prediction, target, mask)
    if isinstance(delta, bool) or not isinstance(delta, Real):
        raise TypeError("delta must be a real number")
    if not math.isfinite(float(delta)) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    valid = mask & torch.isfinite(prediction) & torch.isfinite(target)
    if not valid.any().item():
        return _graph_zero(prediction)
    error = (prediction[valid] - target[valid]).abs()
    delta_tensor = error.new_tensor(float(delta))
    losses = torch.where(
        error < delta_tensor,
        0.5 * error.square(),
        delta_tensor * (error - 0.5 * delta_tensor),
    )
    return losses.mean()


def quantile_pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """Return mean pinball loss over targets and their quantile predictions."""

    if not isinstance(prediction, torch.Tensor) or prediction.ndim < 1:
        raise ValueError("prediction must have a quantile dimension")
    if not isinstance(target, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("target and mask must be torch.Tensor instances")
    if prediction.shape[:-1] != target.shape or mask.shape != target.shape:
        raise ValueError("prediction must have shape target.shape + [Q]")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must have floating-point dtypes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have bool dtype")
    if prediction.device != target.device or mask.device != prediction.device:
        raise ValueError("prediction, target, and mask must be on the same device")
    quantile_values = tuple(quantiles)
    if len(quantile_values) != prediction.shape[-1]:
        raise ValueError("prediction quantile dimension must match quantiles")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
        for value in quantile_values
    ):
        raise ValueError("quantiles must be finite real values in (0, 1)")
    if any(left >= right for left, right in zip(quantile_values, quantile_values[1:])):
        raise ValueError("quantiles must be strictly increasing")

    valid = mask & torch.isfinite(target) & torch.isfinite(prediction).all(dim=-1)
    if not valid.any().item():
        return _graph_zero(prediction)
    errors = target[valid].unsqueeze(-1) - prediction[valid]
    levels = prediction.new_tensor(quantile_values)
    return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()


def negative_cross_sectional_ic_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return one minus mean valid cross-sectional Pearson correlation."""

    _validate_matching_tensors(scores, targets, mask)
    if scores.ndim != 3:
        raise ValueError("scores, targets, and mask must have shape [B, N, H]")
    section_scores = scores.transpose(1, 2)
    section_targets = targets.transpose(1, 2)
    valid = mask.transpose(1, 2)
    valid &= torch.isfinite(section_scores) & torch.isfinite(section_targets)
    counts = valid.sum(dim=-1)
    denominator = counts.clamp_min(1).to(scores.dtype)
    score_mean = torch.where(valid, section_scores, 0.0).sum(dim=-1) / denominator
    target_mean = torch.where(valid, section_targets, 0.0).sum(dim=-1) / denominator
    centered_scores = torch.where(valid, section_scores - score_mean.unsqueeze(-1), 0.0)
    centered_targets = torch.where(valid, section_targets - target_mean.unsqueeze(-1), 0.0)
    score_ss = centered_scores.square().sum(dim=-1)
    target_ss = centered_targets.square().sum(dim=-1)
    eligible = (counts >= 2) & (score_ss > 0) & (target_ss > 0)
    if not eligible.any().item():
        return _graph_zero(scores)
    covariance = (centered_scores * centered_targets).sum(dim=-1)
    correlations = covariance / (score_ss * target_ss).sqrt().clamp_min(
        torch.finfo(scores.dtype).tiny
    )
    return 1.0 - correlations[eligible].mean()


def pairwise_logistic_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    max_pairs: int = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return logistic ranking loss over valid unequal-target asset pairs."""

    _validate_matching_tensors(scores, targets, mask)
    if scores.ndim != 3:
        raise ValueError("scores, targets, and mask must have shape [B, N, H]")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, Integral):
        raise TypeError("max_pairs must be an integer")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    score_differences: list[torch.Tensor] = []
    target_signs: list[torch.Tensor] = []
    for batch_index in range(scores.shape[0]):
        for horizon_index in range(scores.shape[2]):
            section_valid = mask[batch_index, :, horizon_index]
            section_valid &= torch.isfinite(scores[batch_index, :, horizon_index])
            section_valid &= torch.isfinite(targets[batch_index, :, horizon_index])
            valid_indices = section_valid.nonzero(as_tuple=False).squeeze(-1)
            if valid_indices.numel() < 2:
                continue
            pairs = torch.combinations(valid_indices, r=2)
            left, right = pairs[:, 0], pairs[:, 1]
            target_difference = (
                targets[batch_index, left, horizon_index]
                - targets[batch_index, right, horizon_index]
            )
            unequal = target_difference != 0
            if not unequal.any().item():
                continue
            left, right = left[unequal], right[unequal]
            target_difference = target_difference[unequal]
            if left.numel() > max_pairs:
                selected = torch.randperm(
                    left.numel(),
                    device=left.device,
                    generator=generator,
                )[:max_pairs]
                left = left[selected]
                right = right[selected]
                target_difference = target_difference[selected]
            score_differences.append(
                scores[batch_index, left, horizon_index]
                - scores[batch_index, right, horizon_index]
            )
            target_signs.append(target_difference.sign())
    if not score_differences:
        return _graph_zero(scores)

    differences = torch.cat(score_differences)
    signs = torch.cat(target_signs)
    return F.softplus(-signs * differences).mean()


__all__ = [
    "masked_huber_loss",
    "negative_cross_sectional_ic_loss",
    "pairwise_logistic_loss",
    "quantile_pinball_loss",
]
