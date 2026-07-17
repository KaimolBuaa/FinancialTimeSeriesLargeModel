"""Core transformations for Binance USD-M K-line panel data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)


def _timestamps_to_utc(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    unit = "us" if numeric.abs().median() >= 100_000_000_000_000 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def canonicalize_klines(
    raw: pd.DataFrame,
    *,
    symbol: str,
    source_month: str,
) -> pd.DataFrame:
    """Convert one Binance K-line archive into the canonical 1m schema."""

    missing = set(REQUIRED_KLINE_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"missing K-line columns: {sorted(missing)}")

    frame = raw.loc[:, REQUIRED_KLINE_COLUMNS].copy()
    frame["open_time_utc"] = _timestamps_to_utc(frame.pop("open_time"))
    frame["close_time_utc"] = _timestamps_to_utc(frame.pop("close_time"))
    frame = frame.rename(
        columns={
            "volume": "base_volume",
            "taker_buy_base_volume": "taker_buy_base_volume",
            "taker_buy_quote_volume": "taker_buy_quote_volume",
        }
    )
    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame["symbol"] = symbol
    frame["source_month"] = source_month
    frame = frame.sort_values("open_time_utc").drop_duplicates(
        ["symbol", "open_time_utc"], keep="last"
    )
    if frame.empty:
        raise ValueError("K-line archive contains no rows")
    if not np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("K-line archive contains non-finite numeric values")
    if (frame[["base_volume", "quote_volume", "trade_count"]] < 0).any().any():
        raise ValueError("K-line archive contains negative volume or trade count")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("K-line low is above open or close")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("K-line high is below open or close")
    return frame.reset_index(drop=True)


def resample_30m(canonical: pd.DataFrame) -> pd.DataFrame:
    """Aggregate canonical minute K-lines into explicit complete/incomplete bars."""

    required = {
        "open_time_utc",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    }
    missing = required.difference(canonical.columns)
    if missing:
        raise ValueError(f"missing canonical columns: {sorted(missing)}")

    frame = canonical.copy()
    frame["timestamp"] = frame["open_time_utc"].dt.floor("30min")
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "base_volume": "sum",
        "quote_volume": "sum",
        "trade_count": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
        "open_time_utc": "count",
    }
    bars = (
        frame.sort_values(["symbol", "open_time_utc"])
        .groupby(["symbol", "timestamp"], as_index=False)
        .agg(aggregations)
        .rename(columns={"open_time_utc": "minute_count"})
    )
    bars["minute_count"] = bars["minute_count"].astype("int16")
    bars["bar_complete"] = bars["minute_count"].eq(30)
    bars["is_observed"] = bars["minute_count"].gt(0)
    bars["age_bars"] = bars.groupby("symbol").cumcount().astype("int32")
    return bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _future_close(
    frame: pd.DataFrame,
    horizon_bars: int,
) -> tuple[pd.Series, pd.Series]:
    expected = frame["timestamp"] + pd.Timedelta(30 * int(horizon_bars), unit="min")
    lookup = frame.set_index("timestamp")[["close", "bar_complete"]]
    future = lookup.reindex(expected)
    return (
        pd.Series(future["close"].to_numpy(), index=frame.index, dtype=float),
        pd.Series(
            future["bar_complete"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            index=frame.index,
        ),
    )


def add_labels_and_masks(
    bars: pd.DataFrame,
    *,
    history_bars: int = 1_440,
    liquidity_bars: int = 1_440,
    liquidity_threshold: float = 1_000_000.0,
    horizons: Iterable[tuple[str, int]] = (("ret_1d", 48), ("ret_5d", 240)),
) -> pd.DataFrame:
    """Add causal rolling masks and future-return labels to complete 30m bars."""

    if history_bars < 1 or liquidity_bars < 1:
        raise ValueError("history_bars and liquidity_bars must be positive")
    required = {"timestamp", "symbol", "close", "quote_volume", "bar_complete", "is_observed"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"missing 30m columns: {sorted(missing)}")

    parts: list[pd.DataFrame] = []
    for _, part in bars.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        out = part.copy()
        observed = (
            out["is_observed"].astype(bool)
            & out["bar_complete"].astype(bool)
            & np.isfinite(out["close"])
            & np.isfinite(out["quote_volume"])
            & out["close"].gt(0)
            & out["quote_volume"].ge(0)
        )
        out["observed_mask"] = observed
        out["history_mask"] = (
            observed.astype("int8")
            .rolling(history_bars, min_periods=history_bars)
            .sum()
            .eq(history_bars)
        )
        prior_liquidity = out["quote_volume"].where(observed).shift(1)
        out["liquidity_mask"] = (
            prior_liquidity.rolling(liquidity_bars, min_periods=liquidity_bars).mean()
            >= liquidity_threshold
        ).fillna(False)

        label_masks: list[str] = []
        for name, horizon in horizons:
            if horizon < 1:
                raise ValueError("label horizons must be positive")
            future_close, future_complete = _future_close(out, horizon)
            mask_name = f"{name}_mask"
            valid = observed & future_complete & future_close.gt(0)
            out[name] = np.where(valid, np.log(future_close / out["close"]), np.nan)
            out[mask_name] = valid
            label_masks.append(mask_name)

        out["trainable_mask"] = observed & out["history_mask"] & out["liquidity_mask"]
        for mask_name in label_masks:
            out["trainable_mask"] &= out[mask_name]
        parts.append(out)

    return pd.concat(parts, ignore_index=True)
