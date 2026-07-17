"""Tensor contracts for FactorPanel-FM inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_INTEGER_DTYPES.update(
    dtype
    for name in ("uint8", "uint16", "uint32", "uint64")
    if (dtype := getattr(torch, name, None)) is not None
)


@dataclass(frozen=True)
class FactorPanelBatch:
    """A batch of factor panels and their asset/date coordinates."""

    values: torch.Tensor
    observed_mask: torch.Tensor
    asset_ids: torch.Tensor
    dates: torch.Tensor

    def __post_init__(self) -> None:
        tensors = {
            "values": self.values,
            "observed_mask": self.observed_mask,
            "asset_ids": self.asset_ids,
            "dates": self.dates,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if self.values.ndim != 3:
            raise ValueError("values must have shape [B, T, N]")
        if any(dimension == 0 for dimension in self.values.shape):
            raise ValueError("batch dimensions must be nonempty")
        if not self.values.is_floating_point():
            raise TypeError("values must have a floating-point dtype")

        batch_size, context_length, num_assets = self.values.shape
        if self.observed_mask.shape != self.values.shape:
            raise ValueError("observed_mask must have the same shape as values")
        if self.observed_mask.dtype != torch.bool:
            raise TypeError("observed_mask must have bool dtype")
        if self.asset_ids.shape != (batch_size, num_assets):
            raise ValueError("asset_ids must have shape [B, N]")
        if self.dates.shape != (batch_size, context_length):
            raise ValueError("dates must have shape [B, T]")
        if self.asset_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError("asset_ids must have an integer dtype")
        if self.dates.dtype not in _INTEGER_DTYPES:
            raise TypeError("dates must have an integer dtype")
        if not torch.isfinite(self.values[self.observed_mask]).all().item():
            raise ValueError("observed values must be finite")

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def context_length(self) -> int:
        return self.values.shape[1]

    @property
    def num_assets(self) -> int:
        return self.values.shape[2]

    def to(self, device: torch.device | str) -> FactorPanelBatch:
        """Return a new batch with every tensor moved to ``device``."""

        return FactorPanelBatch(
            values=self.values.to(device),
            observed_mask=self.observed_mask.to(device),
            asset_ids=self.asset_ids.to(device),
            dates=self.dates.to(device),
        )
