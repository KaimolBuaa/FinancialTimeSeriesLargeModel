"""Causal 30-minute factor factory for the Binance pretraining panel."""

from __future__ import annotations

import numpy as np
import pandas as pd


TREND_WINDOWS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
FLOW_WINDOWS = TREND_WINDOWS[:6]


def _zscore(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (values - mean) / std


def _symbol_factors(part: pd.DataFrame) -> pd.DataFrame:
    part = part.sort_values("timestamp").copy()
    feature_values: dict[str, pd.Series] = {}
    log_close = np.log(part["close"].where(part["close"] > 0))
    returns = log_close.diff()
    log_quote = np.log1p(part["quote_volume"].clip(lower=0))
    log_trades = np.log1p(part["trade_count"].clip(lower=0))
    true_range = (part["high"] - part["low"]).abs() / part["close"].replace(0, np.nan)
    parkinson = np.log(part["high"] / part["low"]).pow(2)
    imbalance = (
        (2 * part["taker_buy_base_volume"] - part["base_volume"])
        / part["base_volume"].replace(0, np.nan)
    )

    for window in TREND_WINDOWS:
        feature_values[f"factor_return_{window}"] = log_close - log_close.shift(window)
        ema = part["close"].ewm(span=window, adjust=False, min_periods=window).mean()
        feature_values[f"factor_ema_gap_{window}"] = part["close"] / ema - 1
        rolling_high = part["high"].rolling(window, min_periods=window).max()
        rolling_low = part["low"].rolling(window, min_periods=window).min()
        feature_values[f"factor_high_breakout_{window}"] = part["close"] / rolling_high - 1
        feature_values[f"factor_low_breakout_{window}"] = part["close"] / rolling_low - 1

        feature_values[f"factor_realized_vol_{window}"] = returns.rolling(
            window, min_periods=window
        ).std(ddof=0)
        feature_values[f"factor_atr_norm_{window}"] = true_range.rolling(
            window, min_periods=window
        ).mean()
        feature_values[f"factor_parkinson_{window}"] = np.sqrt(
            parkinson.rolling(window, min_periods=window).mean() / (4 * np.log(2))
        )

        feature_values[f"factor_quote_z_{window}"] = _zscore(log_quote, window)
        feature_values[f"factor_trades_z_{window}"] = _zscore(log_trades, window)
        feature_values[f"factor_amihud_{window}"] = (
            returns.abs() / part["quote_volume"].replace(0, np.nan)
        ).rolling(window, min_periods=window).mean()

    for window in FLOW_WINDOWS:
        feature_values[f"factor_quote_change_{window}"] = log_quote - log_quote.shift(window)
        feature_values[f"factor_imbalance_mean_{window}"] = imbalance.rolling(
            window, min_periods=window
        ).mean()
        feature_values[f"factor_imbalance_std_{window}"] = imbalance.rolling(
            window, min_periods=window
        ).std(ddof=0)
        feature_values[f"factor_return_imbalance_corr_{window}"] = returns.rolling(
            window, min_periods=window
        ).corr(imbalance)
    return pd.concat(
        [
            part[["timestamp", "symbol"]].reset_index(drop=True),
            pd.DataFrame(feature_values).reset_index(drop=True),
        ],
        axis=1,
    )


def build_factors(bars: pd.DataFrame) -> pd.DataFrame:
    """Generate exactly 192 causal factor columns from complete/incomplete bars."""

    required = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "minute_count",
        "bar_complete",
        "is_observed",
        "age_bars",
    }
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"missing 30m columns for factors: {sorted(missing)}")

    output = pd.concat(
        [_symbol_factors(part) for _, part in bars.groupby("symbol", sort=False)],
        ignore_index=True,
    ).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # 10 source returns x (cross-sectional rank, z-score, mean deviation) = 30.
    cross_sectional: dict[str, pd.Series] = {}
    for window in TREND_WINDOWS[:10]:
        source = output[f"factor_return_{window}"]
        grouped = output.groupby("timestamp")[f"factor_return_{window}"]
        cross_sectional[f"factor_cs_rank_return_{window}"] = grouped.rank(pct=True)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        cross_sectional[f"factor_cs_z_return_{window}"] = (source - mean) / std
        cross_sectional[f"factor_cs_demean_return_{window}"] = source - mean

    # Six market aggregates x two measures plus six asset/state features = 18.
    for window in TREND_WINDOWS[:6]:
        grouped = output.groupby("timestamp")[f"factor_return_{window}"]
        cross_sectional[f"factor_market_mean_return_{window}"] = grouped.transform("mean")
        cross_sectional[f"factor_market_dispersion_return_{window}"] = grouped.transform("std")
    state = bars.sort_values(["symbol", "timestamp"]).set_index(["symbol", "timestamp"])
    state = state.reindex(pd.MultiIndex.from_frame(output[["symbol", "timestamp"]]))
    cross_sectional["factor_state_log_quote_volume"] = np.log1p(
        state["quote_volume"].clip(lower=0)
    ).to_numpy()
    cross_sectional["factor_state_taker_imbalance"] = (
        (2 * state["taker_buy_base_volume"] - state["base_volume"])
        / state["base_volume"].replace(0, np.nan)
    ).to_numpy()
    cross_sectional["factor_state_log_age"] = np.log1p(state["age_bars"]).to_numpy()
    cross_sectional["factor_state_bar_complete"] = state["bar_complete"].astype("float32").to_numpy()
    cross_sectional["factor_state_minute_fraction"] = (state["minute_count"] / 30).to_numpy()
    cross_sectional["factor_state_observed"] = state["is_observed"].astype("float32").to_numpy()
    output = pd.concat([output, pd.DataFrame(cross_sectional)], axis=1)

    factor_columns = [column for column in output if column.startswith("factor_")]
    if len(factor_columns) != 192:
        raise AssertionError(f"expected 192 factors, produced {len(factor_columns)}")
    return output.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
