"""Column-projected reads and model-shaped panels for ProxyFactor-v0."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


@dataclass(frozen=True)
class ProxyFactorPanel:
    values: np.ndarray
    observed_mask: np.ndarray
    dates: np.ndarray
    assets: tuple[str, ...]
    factor: str


class ProxyFactorStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"complete manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("complete") is not True:
            raise ValueError("ProxyFactor dataset manifest is not complete")
        self.factor_names = frozenset(
            item["name"] if isinstance(item, dict) else str(item)
            for item in self.manifest.get("factors", [])
        )
        if not self.factor_names:
            raise ValueError("ProxyFactor manifest does not define factors")
        factor_root = self.root / "factors"
        if not factor_root.is_dir():
            raise ValueError(f"factor partitions are missing: {factor_root}")
        self._dataset = ds.dataset(
            factor_root,
            format="parquet",
            partitioning="hive",
        )

    def _validate_factor(self, factor: str) -> None:
        if factor not in self.factor_names:
            raise ValueError(f"unknown factor: {factor}")

    def read_factor(
        self,
        factor: str,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        self._validate_factor(factor)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start > end:
            raise ValueError("start_date must not be after end_date")
        date_type = self._dataset.schema.field("date").type
        expression = (ds.field("date") >= pa.scalar(start, type=date_type)) & (
            ds.field("date") <= pa.scalar(end, type=date_type)
        )
        if assets is not None:
            selected = tuple(str(asset) for asset in assets)
            if not selected:
                return pd.DataFrame(columns=["date", "asset", factor])
            expression = expression & ds.field("asset").isin(selected)
        table = self._dataset.to_table(
            columns=["date", "asset", factor],
            filter=expression,
        )
        frame = table.to_pandas()
        if frame.empty:
            return pd.DataFrame(columns=["date", "asset", factor])
        frame["date"] = pd.to_datetime(frame["date"])
        frame["asset"] = frame["asset"].astype(str)
        return frame.sort_values(["date", "asset"], kind="stable").reset_index(
            drop=True
        )

    def read_panel(
        self,
        *,
        factor: str,
        end_date: str | pd.Timestamp,
        context_length: int = 256,
        max_assets: int = 512,
        seed: int = 0,
    ) -> ProxyFactorPanel:
        self._validate_factor(factor)
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 1 <= max_assets <= 512:
            raise ValueError("max_assets must be between 1 and 512")
        end = pd.Timestamp(end_date)
        years = [int(year) for year in self.manifest.get("years", [])]
        if not years:
            raise ValueError("ProxyFactor manifest does not define years")
        earliest = pd.Timestamp(f"{min(years):04d}-01-01")
        span_days = max(context_length * 2, 366)
        start = max(earliest, end - pd.Timedelta(days=span_days))
        while True:
            frame = self.read_factor(
                factor,
                start_date=start,
                end_date=end,
            )
            available_dates = pd.DatetimeIndex(
                frame["date"].drop_duplicates().sort_values()
            )
            if len(available_dates) >= context_length or start == earliest:
                break
            span_days *= 2
            start = max(earliest, end - pd.Timedelta(days=span_days))
        if len(available_dates) < context_length:
            raise ValueError(
                f"only {len(available_dates)} trading dates are available; "
                f"{context_length} required"
            )
        dates = available_dates[-context_length:]
        window = frame.loc[frame["date"].isin(dates)]
        last_date = dates[-1]
        eligible = sorted(
            window.loc[
                (window["date"] == last_date) & window[factor].notna(),
                "asset",
            ].unique()
        )
        if not eligible:
            eligible = sorted(window.loc[window[factor].notna(), "asset"].unique())
        if not eligible:
            raise ValueError("no assets have observed factor values in the window")
        if len(eligible) > max_assets:
            rng = np.random.default_rng(seed)
            selected = sorted(
                str(asset)
                for asset in rng.choice(
                    np.asarray(eligible, dtype=object),
                    size=max_assets,
                    replace=False,
                )
            )
        else:
            selected = eligible
        pivot = window.pivot(index="date", columns="asset", values=factor)
        pivot = pivot.reindex(index=dates, columns=selected)
        observed_mask = pivot.notna().to_numpy(dtype=np.bool_, copy=True)
        values = pivot.fillna(0.0).to_numpy(dtype="float32", copy=True)
        return ProxyFactorPanel(
            values=values,
            observed_mask=observed_mask,
            dates=dates.to_numpy(dtype="datetime64[ns]", copy=True),
            assets=tuple(selected),
            factor=factor,
        )

