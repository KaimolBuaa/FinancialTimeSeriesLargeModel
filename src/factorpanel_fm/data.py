"""Local pandas adapter and tensor sample contracts for FactorPanel-FM.

``PanelFrameDataset`` materializes dense local arrays for smoke tests and
small pilots. It is deliberately not a storage format for a 250k-factor
production corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from .batch import FactorPanelBatch


_FactorIds = str | tuple[str, ...]
_DecisionDates = int | tuple[int, ...]


@dataclass(frozen=True)
class FactorPanelSample:
    """One local dataset sample, or a batch created by ``collate_factor_samples``.

    Dataset items have ``B=1`` with scalar ``factor_id`` and ``decision_date``.
    Collation keeps the same tensor contract while changing those coordinates
    to tuples aligned with the new batch dimension.
    """

    factor_id: _FactorIds
    batch: FactorPanelBatch
    future_factor_targets: torch.Tensor
    future_factor_mask: torch.Tensor
    return_targets: torch.Tensor
    return_mask: torch.Tensor
    decision_date: _DecisionDates

    def __post_init__(self) -> None:
        if not isinstance(self.batch, FactorPanelBatch):
            raise TypeError("batch must be a FactorPanelBatch")
        batch_size = self.batch.batch_size
        if isinstance(self.factor_id, str):
            if not self.factor_id:
                raise ValueError("factor_id must be nonempty")
            if batch_size != 1:
                raise ValueError("scalar factor_id requires batch size 1")
        elif isinstance(self.factor_id, tuple):
            if len(self.factor_id) != batch_size or not all(
                isinstance(value, str) and value for value in self.factor_id
            ):
                raise ValueError(
                    "factor_id tuple must contain one nonempty string per batch row"
                )
        else:
            raise TypeError("factor_id must be a string or tuple of strings")

        if isinstance(self.decision_date, bool):
            raise TypeError("decision_date must be an integer or tuple of integers")
        if isinstance(self.decision_date, Integral):
            if batch_size != 1:
                raise ValueError("scalar decision_date requires batch size 1")
        elif isinstance(self.decision_date, tuple):
            if len(self.decision_date) != batch_size or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in self.decision_date
            ):
                raise ValueError(
                    "decision_date tuple must contain one integer per batch row"
                )
        else:
            raise TypeError("decision_date must be an integer or tuple of integers")

        targets_and_masks = (
            ("future_factor", self.future_factor_targets, self.future_factor_mask),
            ("return", self.return_targets, self.return_mask),
        )
        expected_prefix = (batch_size, self.batch.num_assets)
        batch_tensors = (
            self.batch.values,
            self.batch.observed_mask,
            self.batch.asset_ids,
            self.batch.dates,
        )
        device = self.batch.values.device
        if any(tensor.device != device for tensor in batch_tensors):
            raise ValueError("all batch tensors must be on the same device")
        for name, targets, mask in targets_and_masks:
            if not isinstance(targets, torch.Tensor) or targets.ndim != 3:
                raise ValueError(f"{name}_targets must have shape [B, N, H]")
            if targets.shape[:2] != expected_prefix:
                raise ValueError(f"{name}_targets must have shape [B, N, H]")
            if not targets.is_floating_point():
                raise TypeError(f"{name}_targets must have a floating-point dtype")
            if targets.dtype != self.batch.values.dtype:
                raise TypeError(f"{name}_targets dtype must match batch values")
            if not isinstance(mask, torch.Tensor) or mask.shape != targets.shape:
                raise ValueError(f"{name}_mask must have the same shape as targets")
            if mask.dtype != torch.bool:
                raise TypeError(f"{name}_mask must have bool dtype")
            if targets.device != device or mask.device != device:
                raise ValueError("sample tensors must all be on the same device")
            if not torch.isfinite(targets[mask]).all().item():
                raise ValueError(f"observed {name} targets must be finite")

    def to(self, device: torch.device | str) -> FactorPanelSample:
        """Return a copy with every tensor moved to ``device``."""

        return FactorPanelSample(
            factor_id=self.factor_id,
            batch=self.batch.to(device),
            future_factor_targets=self.future_factor_targets.to(device),
            future_factor_mask=self.future_factor_mask.to(device),
            return_targets=self.return_targets.to(device),
            return_mask=self.return_mask.to(device),
            decision_date=self.decision_date,
        )


def _column_tuple(values: Sequence[str] | None, *, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise TypeError(f"{name} must contain nonempty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _date_codes(values: pd.Index) -> torch.Tensor:
    if pd.api.types.is_integer_dtype(values.dtype):
        return torch.as_tensor(
            values.to_numpy(dtype="int64", copy=True), dtype=torch.int64
        )
    converted = pd.to_datetime(values, errors="raise")
    return torch.as_tensor(
        converted.astype("int64").to_numpy(copy=True), dtype=torch.int64
    )


class PanelFrameDataset(Dataset[FactorPanelSample]):
    """Materialize pandas wide panels into factor-by-window local samples."""

    def __init__(
        self,
        factors: pd.DataFrame,
        labels: pd.DataFrame | None = None,
        *,
        date_col: str = "date",
        asset_col: str = "asset",
        context_length: int = 256,
        future_horizons: Sequence[int] = (5, 20),
        stride: int = 1,
        factor_columns: Sequence[str] | None = None,
        return_columns: Sequence[str] = (),
    ) -> None:
        if not isinstance(factors, pd.DataFrame):
            raise TypeError("factors must be a pandas DataFrame")
        if labels is not None and not isinstance(labels, pd.DataFrame):
            raise TypeError("labels must be a pandas DataFrame or None")
        for name, value in (("context_length", context_length), ("stride", stride)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        horizons = tuple(future_horizons)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            for value in horizons
        ):
            raise ValueError("future_horizons must contain positive integers")
        if len(horizons) != len(set(horizons)):
            raise ValueError("future_horizons must not contain duplicates")
        if not isinstance(date_col, str) or not isinstance(asset_col, str):
            raise TypeError("date_col and asset_col must be strings")
        keys = [date_col, asset_col]
        missing_factor_keys = set(keys).difference(factors.columns)
        if missing_factor_keys:
            raise ValueError(
                f"factors is missing key columns: {sorted(missing_factor_keys)}"
            )
        if factors.duplicated(keys).any():
            raise ValueError("factor date/asset keys must be unique")
        if labels is not None:
            missing_label_keys = set(keys).difference(labels.columns)
            if missing_label_keys:
                raise ValueError(
                    f"labels is missing key columns: {sorted(missing_label_keys)}"
                )
            if labels.duplicated(keys).any():
                raise ValueError("label date/asset keys must be unique")

        selected_factors = _column_tuple(factor_columns, name="factor_columns")
        if factor_columns is None:
            selected_factors = tuple(
                column for column in factors.columns if column not in keys
            )
        selected_returns = _column_tuple(return_columns, name="return_columns")
        if not selected_factors:
            raise ValueError("factor_columns must be nonempty")
        absent_factors = set(selected_factors).difference(factors.columns)
        if absent_factors:
            raise ValueError(f"factors is missing columns: {sorted(absent_factors)}")
        if selected_returns and labels is None:
            raise ValueError("labels is required when return_columns is nonempty")
        absent_returns = set(selected_returns).difference(
            labels.columns if labels is not None else ()
        )
        if absent_returns:
            raise ValueError(f"labels is missing columns: {sorted(absent_returns)}")

        date_values = pd.Index(factors[date_col].drop_duplicates()).sort_values()
        asset_values = pd.Index(factors[asset_col].drop_duplicates()).sort_values()
        grid = pd.MultiIndex.from_product(
            [date_values, asset_values],
            names=[date_col, asset_col],
        )
        factor_grid = factors.set_index(keys).reindex(grid)
        factor_array = factor_grid.loc[:, selected_factors].to_numpy(dtype="float32")
        self._factors = torch.from_numpy(factor_array).reshape(
            len(date_values), len(asset_values), len(selected_factors)
        )
        self._factor_mask = torch.isfinite(self._factors)
        self._factors = torch.where(self._factor_mask, self._factors, 0.0)

        if selected_returns:
            assert labels is not None
            label_grid = labels.set_index(keys).reindex(grid)
            return_array = label_grid.loc[:, selected_returns].to_numpy(dtype="float32")
            self._returns = torch.from_numpy(return_array).reshape(
                len(date_values), len(asset_values), len(selected_returns)
            )
            self._return_mask = torch.isfinite(self._returns)
            self._returns = torch.where(self._return_mask, self._returns, 0.0)
        else:
            self._returns = torch.empty(len(date_values), len(asset_values), 0)
            self._return_mask = torch.empty(
                len(date_values), len(asset_values), 0, dtype=torch.bool
            )

        self.factor_columns = selected_factors
        self.return_columns = selected_returns
        self.future_horizons = tuple(int(value) for value in horizons)
        self.context_length = int(context_length)
        self.stride = int(stride)
        self._integer_dates = pd.api.types.is_integer_dtype(date_values.dtype)
        self.dates = _date_codes(date_values)
        self.asset_ids = torch.arange(len(asset_values), dtype=torch.int64)
        self.asset_values = tuple(asset_values.tolist())
        last_decision = (
            len(date_values) - max(self.future_horizons) - 1
            if self.future_horizons
            else len(date_values) - 1
        )
        self._decision_positions = tuple(
            range(self.context_length - 1, last_decision + 1, self.stride)
        )

    @classmethod
    def from_parquet(
        cls,
        factors_path: str | Path,
        labels_path: str | Path | None = None,
        **kwargs: object,
    ) -> PanelFrameDataset:
        """Read parquet frames and construct the local materializing adapter."""

        factors = pd.read_parquet(factors_path)
        labels = pd.read_parquet(labels_path) if labels_path is not None else None
        return cls(factors, labels, **kwargs)

    def __len__(self) -> int:
        return len(self.factor_columns) * len(self._decision_positions)

    def __getitem__(self, index: int) -> FactorPanelSample:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        windows_per_factor = len(self._decision_positions)
        factor_index, window_index = divmod(int(index), windows_per_factor)
        decision_position = self._decision_positions[window_index]
        start = decision_position - self.context_length + 1
        values = self._factors[start : decision_position + 1, :, factor_index]
        observed = self._factor_mask[start : decision_position + 1, :, factor_index]
        if self.future_horizons:
            future_positions = [
                decision_position + horizon for horizon in self.future_horizons
            ]
            future = self._factors[future_positions, :, factor_index].transpose(0, 1)
            future_mask = self._factor_mask[
                future_positions, :, factor_index
            ].transpose(0, 1)
        else:
            future = torch.empty(self.asset_ids.numel(), 0, dtype=self._factors.dtype)
            future_mask = torch.empty(self.asset_ids.numel(), 0, dtype=torch.bool)
        returns = self._returns[decision_position]
        return_mask = self._return_mask[decision_position]
        decision_date = int(self.dates[decision_position].item())
        return FactorPanelSample(
            factor_id=self.factor_columns[factor_index],
            batch=FactorPanelBatch(
                values=values.unsqueeze(0),
                observed_mask=observed.unsqueeze(0),
                asset_ids=self.asset_ids.unsqueeze(0),
                dates=self.dates[start : decision_position + 1].unsqueeze(0),
            ),
            future_factor_targets=future.unsqueeze(0),
            future_factor_mask=future_mask.unsqueeze(0),
            return_targets=returns.unsqueeze(0),
            return_mask=return_mask.unsqueeze(0),
            decision_date=decision_date,
        )


def collate_factor_samples(samples: Sequence[FactorPanelSample]) -> FactorPanelSample:
    """Stack singleton dataset samples after coordinate compatibility checks."""

    items = tuple(samples)
    if not items:
        raise ValueError("samples must be nonempty")
    if any(not isinstance(sample, FactorPanelSample) for sample in items):
        raise TypeError("samples must contain FactorPanelSample instances")
    if any(sample.batch.batch_size != 1 for sample in items):
        raise ValueError("collate_factor_samples accepts singleton samples only")
    first = items[0]
    for sample in items[1:]:
        if not torch.equal(sample.batch.asset_ids, first.batch.asset_ids):
            raise ValueError("asset_ids must be identical across samples")
        if sample.batch.context_length != first.batch.context_length:
            raise ValueError("context lengths must match")
        if (
            sample.future_factor_targets.shape[1:]
            != first.future_factor_targets.shape[1:]
        ):
            raise ValueError("future target shapes must match")
        if sample.return_targets.shape[1:] != first.return_targets.shape[1:]:
            raise ValueError("return target shapes must match")
        if sample.batch.values.dtype != first.batch.values.dtype:
            raise ValueError("sample dtypes must match")
        if sample.batch.values.device != first.batch.values.device:
            raise ValueError("sample devices must match")

    def stack(name: str) -> torch.Tensor:
        return torch.cat([getattr(sample, name) for sample in items], dim=0)

    return FactorPanelSample(
        factor_id=tuple(str(sample.factor_id) for sample in items),
        batch=FactorPanelBatch(
            values=torch.cat([sample.batch.values for sample in items], dim=0),
            observed_mask=torch.cat(
                [sample.batch.observed_mask for sample in items], dim=0
            ),
            asset_ids=torch.cat([sample.batch.asset_ids for sample in items], dim=0),
            dates=torch.cat([sample.batch.dates for sample in items], dim=0),
        ),
        future_factor_targets=stack("future_factor_targets"),
        future_factor_mask=stack("future_factor_mask"),
        return_targets=stack("return_targets"),
        return_mask=stack("return_mask"),
        decision_date=tuple(int(sample.decision_date) for sample in items),
    )


def _boundary_code(value: object, dataset: PanelFrameDataset) -> int:
    if isinstance(value, bool):
        raise TypeError("date boundaries must not be bool")
    if isinstance(value, Integral):
        return int(value)
    if dataset._integer_dates:
        timestamp = pd.Timestamp(value)
        return int(timestamp.strftime("%Y%m%d"))
    return int(_date_codes(pd.Index([value]))[0].item())


def chronological_split_indices(
    dataset: PanelFrameDataset,
    train_end: object,
    valid_end: object,
    purge: int = 20,
) -> tuple[list[int], list[int], list[int]]:
    """Split samples by decision-date positions, purging earlier split tails."""

    if not isinstance(dataset, PanelFrameDataset):
        raise TypeError("dataset must be a PanelFrameDataset")
    if isinstance(purge, bool) or not isinstance(purge, Integral):
        raise TypeError("purge must be an integer")
    if purge < 0:
        raise ValueError("purge must be nonnegative")
    train_code = _boundary_code(train_end, dataset)
    valid_code = _boundary_code(valid_end, dataset)
    if train_code >= valid_code:
        raise ValueError("train_end must be before valid_end")
    dates = dataset.dates.tolist()
    train_boundary = max(
        (index for index, date in enumerate(dates) if date <= train_code), default=-1
    )
    valid_boundary = max(
        (index for index, date in enumerate(dates) if date <= valid_code), default=-1
    )
    train_cutoff = train_boundary - int(purge)
    valid_cutoff = valid_boundary - int(purge)
    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    windows = len(dataset._decision_positions)
    for factor_index in range(len(dataset.factor_columns)):
        for window_index, position in enumerate(dataset._decision_positions):
            sample_index = factor_index * windows + window_index
            if position <= train_cutoff:
                train.append(sample_index)
            elif train_boundary < position <= valid_cutoff:
                valid.append(sample_index)
            elif position > valid_boundary:
                test.append(sample_index)
    return train, valid, test


__all__ = [
    "FactorPanelSample",
    "PanelFrameDataset",
    "chronological_split_indices",
    "collate_factor_samples",
]
