"""Causal input views derived from factor panels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .batch import FactorPanelBatch


@dataclass(frozen=True)
class InputViews:
    """Rank, robust-scale, and observation channels for a factor panel."""

    rank_gaussian: torch.Tensor
    robust_z: torch.Tensor
    observed_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.rank_gaussian.ndim != 3:
            raise ValueError("input views must have shape [B, T, N]")
        if self.robust_z.shape != self.rank_gaussian.shape:
            raise ValueError("robust_z must match rank_gaussian shape")
        if self.observed_mask.shape != self.rank_gaussian.shape:
            raise ValueError("observed_mask must match rank_gaussian shape")
        if self.observed_mask.dtype != torch.bool:
            raise TypeError("observed_mask must have bool dtype")

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


def _rank_gaussian(
    values: torch.Tensor,
    observed_mask: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    output = torch.zeros_like(values)
    for batch_index in range(values.shape[0]):
        for time_index in range(values.shape[1]):
            valid = observed_mask[batch_index, time_index]
            count = int(valid.sum().item())
            if count == 0:
                continue

            current = values[batch_index, time_index, valid].to(torch.float64)
            _, inverse, counts = torch.unique(
                current,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            starts = torch.cumsum(counts, dim=0) - counts
            midranks = starts.to(torch.float64) + (counts.to(torch.float64) + 1.0) / 2.0
            percentiles = (midranks[inverse] - 0.5) / count
            transformed = math.sqrt(2.0) * torch.erfinv(2.0 * percentiles - 1.0)
            output[batch_index, time_index, valid] = transformed.clamp(-clip, clip).to(
                output.dtype
            )
    return output


def _median(values: torch.Tensor) -> torch.Tensor:
    ordered = values.sort().values
    midpoint = ordered.numel() // 2
    if ordered.numel() % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _robust_z(
    values: torch.Tensor,
    observed_mask: torch.Tensor,
    window: int,
    clip: float,
    eps: float,
) -> torch.Tensor:
    output = torch.zeros_like(values)
    for batch_index in range(values.shape[0]):
        for asset_index in range(values.shape[2]):
            for time_index in range(values.shape[1]):
                if not observed_mask[batch_index, time_index, asset_index].item():
                    continue
                start = max(0, time_index - window)
                history_mask = observed_mask[
                    batch_index, start:time_index, asset_index
                ]
                if not history_mask.any().item():
                    continue
                history = values[
                    batch_index, start:time_index, asset_index
                ][history_mask].to(torch.float64)
                center = _median(history)
                scale = _median((history - center).abs()) * 1.4826
                if scale.item() <= eps:
                    continue
                current = values[batch_index, time_index, asset_index].to(torch.float64)
                score = ((current - center) / scale).clamp(-clip, clip)
                output[batch_index, time_index, asset_index] = score.to(output.dtype)
    return output


def build_input_views(
    batch: FactorPanelBatch,
    robust_window: int = 252,
    rank_clip: float = 3.0,
    z_clip: float = 5.0,
    eps: float = 1e-6,
) -> InputViews:
    """Build permutation-equivariant cross-sectional and causal time-series views."""

    if not isinstance(robust_window, int) or robust_window <= 0:
        raise ValueError("robust_window must be a positive integer")
    if rank_clip < 0 or z_clip < 0:
        raise ValueError("clip values must be nonnegative")
    if eps <= 0:
        raise ValueError("eps must be positive")

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
