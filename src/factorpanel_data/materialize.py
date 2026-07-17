"""Read verified raw archives and write each deterministic training-data layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .factors import build_factors
from .pipeline import REQUIRED_KLINE_COLUMNS, add_labels_and_masks, canonicalize_klines, resample_30m


SOURCE_COLUMNS = (*REQUIRED_KLINE_COLUMNS, "ignore")


def read_archive(archive: Path) -> pd.DataFrame:
    """Read one headerless Binance monthly ZIP without extracting it to disk."""

    return pd.read_csv(
        archive,
        compression="zip",
        header=None,
        names=SOURCE_COLUMNS,
        usecols=range(len(SOURCE_COLUMNS)),
    )


def _partition_path(data_root: Path, layer: str, symbol: str, source_month: str) -> Path:
    return data_root / layer / f"symbol={symbol}" / f"year={source_month[:4]}" / f"{source_month}.parquet"


def materialize_archive(
    *,
    archive: Path,
    data_root: Path,
    symbol: str,
    source_month: str,
    history_bars: int = 1_440,
    liquidity_bars: int = 1_440,
    liquidity_threshold: float = 1_000_000.0,
) -> dict[str, Path]:
    """Materialize canonical, 30m, factor, and label/mask layers for one archive."""

    canonical = canonicalize_klines(
        read_archive(archive),
        symbol=symbol,
        source_month=source_month,
    )
    bars = resample_30m(canonical)
    factors = build_factors(bars)
    labels = add_labels_and_masks(
        bars,
        history_bars=history_bars,
        liquidity_bars=liquidity_bars,
        liquidity_threshold=liquidity_threshold,
    )

    outputs = {
        "canonical": _partition_path(data_root, "canonical_1m", symbol, source_month),
        "train_30m": _partition_path(data_root, "train_30m", symbol, source_month),
        "factors": _partition_path(data_root, "factors_30m", symbol, source_month),
        "targets_masks": _partition_path(data_root, "targets_masks_30m", symbol, source_month),
    }
    factors = factors[["timestamp", "symbol", *[c for c in factors if c.startswith("factor_")]]]
    labels = labels[
        [
            "timestamp",
            "symbol",
            "ret_1d",
            "ret_5d",
            "observed_mask",
            "history_mask",
            "liquidity_mask",
            "ret_1d_mask",
            "ret_5d_mask",
            "trainable_mask",
        ]
    ]
    for key, frame in (
        ("canonical", canonical),
        ("train_30m", bars),
        ("factors", factors),
        ("targets_masks", labels),
    ):
        outputs[key].parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(outputs[key], index=False)
    return outputs


def materialize_archives(
    *,
    archives: list[tuple[Path, str]],
    data_root: Path,
    symbol: str,
    history_bars: int = 1_440,
    liquidity_bars: int = 1_440,
    liquidity_threshold: float = 1_000_000.0,
) -> dict[str, Path]:
    """Materialize a contiguous symbol history so windows survive month boundaries."""

    if not archives:
        raise ValueError("archives must not be empty")
    source_months = [month for _, month in archives]
    canonical = pd.concat(
        [
            canonicalize_klines(read_archive(archive), symbol=symbol, source_month=month)
            for archive, month in archives
        ],
        ignore_index=True,
    )
    canonical = canonical.sort_values("open_time_utc").drop_duplicates(
        ["symbol", "open_time_utc"], keep="last"
    ).reset_index(drop=True)
    bars = resample_30m(canonical)
    factors = build_factors(bars)
    labels = add_labels_and_masks(
        bars,
        history_bars=history_bars,
        liquidity_bars=liquidity_bars,
        liquidity_threshold=liquidity_threshold,
    )
    build_name = f"{min(source_months)}_to_{max(source_months)}"
    outputs = {
        "canonical": data_root / "canonical_1m" / f"symbol={symbol}" / f"build={build_name}.parquet",
        "train_30m": data_root / "train_30m" / f"symbol={symbol}" / f"build={build_name}.parquet",
        "factors": data_root / "factors_30m" / f"symbol={symbol}" / f"build={build_name}.parquet",
        "targets_masks": data_root / "targets_masks_30m" / f"symbol={symbol}" / f"build={build_name}.parquet",
    }
    factor_frame = factors[
        ["timestamp", "symbol", *[column for column in factors if column.startswith("factor_")]]
    ]
    label_frame = labels[
        [
            "timestamp",
            "symbol",
            "ret_1d",
            "ret_5d",
            "observed_mask",
            "history_mask",
            "liquidity_mask",
            "ret_1d_mask",
            "ret_5d_mask",
            "trainable_mask",
        ]
    ]
    for key, frame in (
        ("canonical", canonical),
        ("train_30m", bars),
        ("factors", factor_frame),
        ("targets_masks", label_frame),
    ):
        outputs[key].parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(outputs[key], index=False)
    return outputs


def materialize_panel(
    *,
    archives_by_symbol: dict[str, list[tuple[Path, str]]],
    data_root: Path,
    history_bars: int = 1_440,
    liquidity_bars: int = 1_440,
    liquidity_threshold: float = 1_000_000.0,
) -> dict[str, Path]:
    """Build one multi-asset panel so cross-sectional factors share each timestamp."""

    if not archives_by_symbol:
        raise ValueError("archives_by_symbol must not be empty")
    canonical_parts: list[pd.DataFrame] = []
    source_months: list[str] = []
    for symbol, archives in archives_by_symbol.items():
        if not archives:
            continue
        for archive, month in archives:
            canonical_parts.append(
                canonicalize_klines(read_archive(archive), symbol=symbol, source_month=month)
            )
            source_months.append(month)
    if not canonical_parts:
        raise ValueError("archives_by_symbol did not contain any archives")

    canonical = pd.concat(canonical_parts, ignore_index=True)
    canonical = canonical.sort_values(["symbol", "open_time_utc"]).drop_duplicates(
        ["symbol", "open_time_utc"], keep="last"
    ).reset_index(drop=True)
    bars = resample_30m(canonical)
    factors = build_factors(bars)
    labels = add_labels_and_masks(
        bars,
        history_bars=history_bars,
        liquidity_bars=liquidity_bars,
        liquidity_threshold=liquidity_threshold,
    )
    build_name = f"{min(source_months)}_to_{max(source_months)}"
    outputs = {
        "canonical": data_root / "canonical_1m" / f"build={build_name}.parquet",
        "train_30m": data_root / "train_30m" / f"build={build_name}.parquet",
        "factors": data_root / "factors_30m" / f"build={build_name}.parquet",
        "targets_masks": data_root / "targets_masks_30m" / f"build={build_name}.parquet",
    }
    factor_frame = factors[
        ["timestamp", "symbol", *[column for column in factors if column.startswith("factor_")]]
    ]
    label_frame = labels[
        [
            "timestamp",
            "symbol",
            "ret_1d",
            "ret_5d",
            "observed_mask",
            "history_mask",
            "liquidity_mask",
            "ret_1d_mask",
            "ret_5d_mask",
            "trainable_mask",
        ]
    ]
    for key, frame in (
        ("canonical", canonical),
        ("train_30m", bars),
        ("factors", factor_frame),
        ("targets_masks", label_frame),
    ):
        outputs[key].parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(outputs[key], index=False)
    return outputs
