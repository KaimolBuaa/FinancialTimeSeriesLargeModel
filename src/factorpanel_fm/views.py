"""Causal input views derived from factor panels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch

from .batch import FactorPanelBatch


@dataclass(frozen=True)
class InputViews:
    """Rank, robust-scale, and observation channels for a factor panel."""

    rank_gaussian: torch.Tensor
    robust_z: torch.Tensor
    observed_mask: torch.Tensor

    def __post_init__(self) -> None:
        tensors = {
            "rank_gaussian": self.rank_gaussian,
            "robust_z": self.robust_z,
            "observed_mask": self.observed_mask,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if self.rank_gaussian.ndim != 3:
            raise ValueError("input views must have shape [B, T, N]")
        if self.robust_z.shape != self.rank_gaussian.shape:
            raise ValueError("robust_z must match rank_gaussian shape")
        if self.observed_mask.shape != self.rank_gaussian.shape:
            raise ValueError("observed_mask must match rank_gaussian shape")
        if not self.rank_gaussian.is_floating_point():
            raise TypeError("rank_gaussian must have a floating-point dtype")
        if not self.robust_z.is_floating_point():
            raise TypeError("robust_z must have a floating-point dtype")
        if self.robust_z.dtype != self.rank_gaussian.dtype:
            raise TypeError("numeric input views must have the same dtype")
        if self.robust_z.device != self.rank_gaussian.device:
            raise ValueError("numeric input views must be on the same device")
        if self.observed_mask.dtype != torch.bool:
            raise TypeError("observed_mask must have bool dtype")
        if self.observed_mask.device != self.rank_gaussian.device:
            raise ValueError("all input views must be on the same device")

    @property
    def stacked(self) -> torch.Tensor:
        """Return the three views as floating channels in the last dimension."""

        return torch.stack(
            (
                self.rank_gaussian,
                self.robust_z,
                self.observed_mask.to(dtype=self.rank_gaussian.dtype),
            ),
            dim=-1,
        )


def _working_dtype(values: torch.Tensor) -> torch.dtype:
    if values.device.type == "mps":
        return torch.float32
    return torch.float64


def _rank_gaussian(
    values: torch.Tensor,
    observed_mask: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    working = values.to(_working_dtype(values))
    infinity = torch.full((), float("inf"), dtype=working.dtype, device=working.device)
    sortable = torch.where(observed_mask, working, infinity)
    ordered, order = sortable.sort(dim=-1)

    num_assets = values.shape[-1]
    positions = torch.arange(num_assets, device=values.device)
    counts = observed_mask.sum(dim=-1)
    valid_ordered = positions < counts.unsqueeze(-1)

    starts = torch.ones_like(valid_ordered)
    starts[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    starts &= valid_ordered
    start_positions = torch.where(starts, positions, 0)
    group_starts = start_positions.cummax(dim=-1).values

    ends = torch.ones_like(valid_ordered)
    ends[..., :-1] = ordered[..., :-1] != ordered[..., 1:]
    ends &= valid_ordered
    end_positions = torch.where(ends, positions, num_assets - 1)
    group_ends = end_positions.flip(-1).cummin(dim=-1).values.flip(-1)

    numerator = (group_starts + group_ends + 1).to(working.dtype)
    denominator = 2.0 * counts.clamp_min(1).unsqueeze(-1).to(working.dtype)
    percentiles = torch.where(valid_ordered, numerator / denominator, 0.5)
    transformed = math.sqrt(2.0) * torch.erfinv(2.0 * percentiles - 1.0)
    clip_value = min(
        clip,
        torch.finfo(working.dtype).max,
        torch.finfo(values.dtype).max,
    )
    transformed = torch.where(
        valid_ordered,
        transformed.clamp(-clip_value, clip_value),
        0.0,
    )
    return torch.zeros_like(working).scatter(-1, order, transformed).to(values.dtype)


def _ordered_median(ordered: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    lower_indices = ((counts - 1).clamp_min(0) // 2).unsqueeze(-1)
    upper_indices = (counts // 2).unsqueeze(-1)
    lower = ordered.gather(-1, lower_indices).squeeze(-1)
    upper = ordered.gather(-1, upper_indices).squeeze(-1)
    midpoint = lower / 2.0 + upper / 2.0
    return torch.where(counts.remainder(2) == 1, lower, midpoint)


def _robust_z(
    values: torch.Tensor,
    observed_mask: torch.Tensor,
    window: int,
    clip: float,
    eps: float,
) -> torch.Tensor:
    working = values.to(_working_dtype(values))
    safe_values = torch.where(observed_mask, working, 0.0)
    effective_window = min(window, values.shape[1])
    value_padding = torch.zeros(
        values.shape[0],
        effective_window,
        values.shape[2],
        dtype=working.dtype,
        device=values.device,
    )
    mask_padding = torch.zeros(
        values.shape[0],
        effective_window,
        values.shape[2],
        dtype=torch.bool,
        device=values.device,
    )
    history_values = torch.cat((value_padding, safe_values), dim=1).unfold(
        1, effective_window, 1
    )[:, : values.shape[1]]
    history_masks = torch.cat((mask_padding, observed_mask), dim=1).unfold(
        1, effective_window, 1
    )[:, : values.shape[1]]

    infinity = torch.full((), float("inf"), dtype=working.dtype, device=working.device)
    clip_value = min(
        clip,
        torch.finfo(working.dtype).max,
        torch.finfo(values.dtype).max,
    )
    eps_value = min(eps, torch.finfo(working.dtype).max)
    chunks = []
    chunk_size = 16
    for start in range(0, values.shape[1], chunk_size):
        stop = min(start + chunk_size, values.shape[1])
        chunk_values = history_values[:, start:stop]
        chunk_masks = history_masks[:, start:stop]
        counts = chunk_masks.sum(dim=-1)

        normalizer = torch.where(chunk_masks, chunk_values.abs(), 0.0).amax(dim=-1)
        normalizer = normalizer.clamp_min(1.0)
        normalized = chunk_values / normalizer.unsqueeze(-1)
        ordered = torch.where(chunk_masks, normalized, infinity).sort(dim=-1).values
        center = _ordered_median(ordered, counts)

        deviations = (normalized - center.unsqueeze(-1)).abs()
        ordered_deviations = torch.where(chunk_masks, deviations, infinity).sort(
            dim=-1
        ).values
        scale = _ordered_median(ordered_deviations, counts) * 1.4826

        current = working[:, start:stop] / normalizer
        safe_scale = torch.where(scale > 0, scale, 1.0)
        scores = ((current - center) / safe_scale).clamp(-clip_value, clip_value)
        usable = (
            observed_mask[:, start:stop]
            & (counts > 0)
            & (scale > eps_value / normalizer)
        )
        chunks.append(torch.where(usable, scores, 0.0))

    return torch.cat(chunks, dim=1).to(values.dtype)


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def build_input_views(
    batch: FactorPanelBatch,
    robust_window: int = 252,
    rank_clip: float = 3.0,
    z_clip: float = 5.0,
    eps: float = 1e-6,
) -> InputViews:
    """Build permutation-equivariant cross-sectional and causal time-series views."""

    if (
        isinstance(robust_window, bool)
        or not isinstance(robust_window, Integral)
        or robust_window <= 0
    ):
        raise ValueError("robust_window must be a positive integer")
    robust_window = int(robust_window)
    rank_clip = _positive_real("rank_clip", rank_clip)
    z_clip = _positive_real("z_clip", z_clip)
    eps = _positive_real("eps", eps)

    return InputViews(
        rank_gaussian=_rank_gaussian(batch.values, batch.observed_mask, rank_clip),
        robust_z=_robust_z(
            batch.values,
            batch.observed_mask,
            robust_window,
            z_clip,
            eps,
        ),
        observed_mask=batch.observed_mask,
    )
