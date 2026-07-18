"""Quality gates and final manifest publication for ProxyFactor-v0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import json
import os
from pathlib import Path
from typing import Iterable, Sequence
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .proxy_config import ProxyFactorConfig
from .proxy_materialize import normalize_proxy_frame, sha256_file
from .proxy_registry import (
    FACTOR_WINDOWS,
    PROXY_FACTOR_REGISTRY,
    PROXY_LABEL_REGISTRY,
)
from .qlib_proxy import ProxyProvider, query_factor_range


FORMULA_AUDIT_FACTORS = (
    "pf_kmid",
    "pf_klen",
    "pf_kmid2",
    "pf_open_close",
    "pf_high_close",
    "pf_low_close",
    "pf_vwap_close",
    "pf_return_1",
    "pf_volume_change_1",
    "pf_amount_change_1",
    "pf_roc_20",
    "pf_ma_20",
    "pf_std_20",
    "pf_beta_20",
    "pf_rsqr_20",
    "pf_max_20",
    "pf_min_20",
    "pf_rsv_20",
    "pf_corr_20",
    "pf_cord_20",
    "pf_cntd_20",
    "pf_sumd_20",
    "pf_vma_20",
    "pf_vstd_20",
)
RAW_AUDIT_FIELDS = ("open", "high", "low", "close", "vwap", "volume", "amount")


@dataclass(frozen=True)
class FactorQualityStats:
    valid_count: int
    total_count: int
    valid_ratio: float
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    near_constant_ratio: float
    nonfinite_count: int


@dataclass(frozen=True)
class PartitionQualityReport:
    rows: int
    duplicate_keys: int
    nonfinite_values: int
    sorted_keys: bool
    factors: dict[str, FactorQualityStats]


def _factor_stats(values: np.ndarray) -> FactorQualityStats:
    array = np.asarray(values, dtype="float32")
    finite_mask = np.isfinite(array)
    finite = array[finite_mask]
    valid_count = int(finite.size)
    total_count = int(array.size)
    nonfinite_count = int(np.isinf(array).sum())
    if valid_count == 0:
        return FactorQualityStats(
            valid_count=0,
            total_count=total_count,
            valid_ratio=0.0,
            mean=None,
            std=None,
            minimum=None,
            maximum=None,
            near_constant_ratio=0.0,
            nonfinite_count=nonfinite_count,
        )
    _, counts = np.unique(finite, return_counts=True)
    return FactorQualityStats(
        valid_count=valid_count,
        total_count=total_count,
        valid_ratio=valid_count / total_count if total_count else 0.0,
        mean=float(np.mean(finite, dtype="float64")),
        std=float(np.std(finite, dtype="float64", ddof=0)),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        near_constant_ratio=float(counts.max() / valid_count),
        nonfinite_count=nonfinite_count,
    )


def inspect_factor_partition(
    frame: pd.DataFrame,
    expected_factors: Sequence[str],
) -> PartitionQualityReport:
    missing = [name for name in expected_factors if name not in frame]
    if missing:
        raise ValueError(f"factor partition is missing columns: {missing}")
    if not {"date", "asset"}.issubset(frame.columns):
        raise ValueError("factor partition must contain date and asset")
    duplicate_keys = int(frame.duplicated(["date", "asset"]).sum())
    key_index = pd.MultiIndex.from_frame(frame[["date", "asset"]])
    factors = {
        name: _factor_stats(frame[name].to_numpy(copy=False))
        for name in expected_factors
    }
    return PartitionQualityReport(
        rows=len(frame),
        duplicate_keys=duplicate_keys,
        nonfinite_values=sum(item.nonfinite_count for item in factors.values()),
        sorted_keys=key_index.is_monotonic_increasing,
        factors=factors,
    )


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype="float64")
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return series.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(values - values.mean(), centered) / denominator),
        raw=True,
    )


def compute_proxy_factors_pandas(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "vwap", "volume", "amount"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"raw K-line frame is missing columns: {missing}")
    open_ = raw["open"].astype("float64")
    high = raw["high"].astype("float64")
    low = raw["low"].astype("float64")
    close = raw["close"].astype("float64")
    vwap = raw["vwap"].astype("float64")
    volume = raw["volume"].astype("float64")
    amount = raw["amount"].astype("float64")
    epsilon = 1e-12
    result: dict[str, pd.Series] = {
        "pf_kmid": (close - open_) / (open_ + epsilon),
        "pf_klen": (high - low) / (open_ + epsilon),
        "pf_kmid2": (close - open_) / (high - low + epsilon),
        "pf_kup": (high - np.maximum(open_, close)) / (open_ + epsilon),
        "pf_kup2": (high - np.maximum(open_, close)) / (high - low + epsilon),
        "pf_klow": (np.minimum(open_, close) - low) / (open_ + epsilon),
        "pf_klow2": (np.minimum(open_, close) - low) / (high - low + epsilon),
        "pf_ksft": (2 * close - high - low) / (open_ + epsilon),
        "pf_ksft2": (2 * close - high - low) / (high - low + epsilon),
        "pf_open_close": open_ / (close + epsilon) - 1,
        "pf_high_close": high / (close + epsilon) - 1,
        "pf_low_close": low / (close + epsilon) - 1,
        "pf_vwap_close": vwap / (close + epsilon) - 1,
        "pf_return_1": close / (close.shift(1) + epsilon) - 1,
        "pf_volume_change_1": volume / (volume.shift(1) + epsilon) - 1,
        "pf_amount_change_1": amount / (amount.shift(1) + epsilon) - 1,
    }
    close_return = close / (close.shift(1) + epsilon) - 1
    volume_log_change = np.log(volume / (volume.shift(1) + epsilon) + 1)
    up = (close > close.shift(1)).astype("float64")
    down = (close < close.shift(1)).astype("float64")
    delta = close - close.shift(1)
    positive_delta = delta.clip(lower=0)
    negative_delta = (-delta).clip(lower=0)
    time = pd.Series(np.arange(len(close), dtype="float64"), index=close.index)
    for window in FACTOR_WINDOWS:
        rolling_close = close.rolling(window, min_periods=window)
        rolling_volume = volume.rolling(window, min_periods=window)
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        result[f"pf_roc_{window}"] = close / (close.shift(window) + epsilon) - 1
        result[f"pf_ma_{window}"] = close / (rolling_close.mean() + epsilon) - 1
        result[f"pf_std_{window}"] = rolling_close.std() / (close + epsilon)
        result[f"pf_beta_{window}"] = _rolling_slope(close, window) / (close + epsilon)
        rsquared_window = max(window, 3)
        result[f"pf_rsqr_{window}"] = (
            close.rolling(
                rsquared_window,
                min_periods=rsquared_window,
            )
            .corr(time)
            .pow(2)
        )
        result[f"pf_max_{window}"] = close / (rolling_high + epsilon) - 1
        result[f"pf_min_{window}"] = close / (rolling_low + epsilon) - 1
        result[f"pf_rsv_{window}"] = (close - rolling_low) / (
            rolling_high - rolling_low + epsilon
        )
        result[f"pf_corr_{window}"] = rolling_close.corr(np.log(volume + 1))
        result[f"pf_cord_{window}"] = close_return.rolling(
            window, min_periods=window
        ).corr(volume_log_change)
        result[f"pf_cntd_{window}"] = (
            up.rolling(window, min_periods=window).mean()
            - down.rolling(window, min_periods=window).mean()
        )
        result[f"pf_sumd_{window}"] = (
            positive_delta.rolling(window, min_periods=window).sum()
            - negative_delta.rolling(window, min_periods=window).sum()
        ) / (delta.abs().rolling(window, min_periods=window).sum() + epsilon)
        result[f"pf_vma_{window}"] = volume / (rolling_volume.mean() + epsilon) - 1
        result[f"pf_vstd_{window}"] = rolling_volume.std() / (
            rolling_volume.mean() + epsilon
        )
    ordered_names = [item.name for item in PROXY_FACTOR_REGISTRY]
    return pd.DataFrame(result, index=raw.index).loc[:, ordered_names]


def verify_pandas_causality(
    *,
    seed: int = 7,
    periods: int = 260,
    cutoff: int = 200,
) -> dict:
    if periods <= 120 or not 120 <= cutoff < periods - 1:
        raise ValueError("causality check requires 120 <= cutoff < periods - 1")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    open_ = close * np.exp(rng.normal(0.0, 0.003, periods))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.02, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.02, periods))
    volume = rng.lognormal(mean=12.0, sigma=0.5, size=periods)
    vwap = (open_ + high + low + close) / 4
    raw = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vwap": vwap,
            "volume": volume,
            "amount": vwap * volume,
        },
        index=pd.date_range("2000-01-03", periods=periods, freq="B"),
    )
    original = compute_proxy_factors_pandas(raw)
    perturbed = raw.copy()
    future = perturbed.index[cutoff + 1 :]
    price_columns = ["open", "high", "low", "close", "vwap"]
    perturbed.loc[future, price_columns] *= rng.uniform(
        0.5,
        1.5,
        size=(len(future), 1),
    )
    perturbed.loc[future, "volume"] *= rng.uniform(0.2, 3.0, len(future))
    perturbed.loc[future, "amount"] = (
        perturbed.loc[future, "vwap"] * perturbed.loc[future, "volume"]
    )
    recomputed = compute_proxy_factors_pandas(perturbed)
    try:
        np.testing.assert_allclose(
            original.iloc[: cutoff + 1].to_numpy(),
            recomputed.iloc[: cutoff + 1].to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    except AssertionError as error:
        return {
            "passed": False,
            "factors_checked": original.shape[1],
            "error": str(error),
        }
    return {"passed": True, "factors_checked": original.shape[1]}


def _default_audit_years(config: ProxyFactorConfig) -> tuple[int, ...]:
    years = config.years
    if len(years) < 3:
        return years
    return years[0], years[len(years) // 2], years[-1]


def verify_qlib_formula_audit(
    config: ProxyFactorConfig,
    provider: ProxyProvider,
    *,
    years: Sequence[int] | None = None,
    max_assets: int = 3,
    rtol: float = 2e-4,
    atol: float = 2e-5,
) -> dict:
    """Compare registered Qlib expressions with an independent pandas calculation."""
    selected_years = tuple(years) if years is not None else _default_audit_years(config)
    required_year_count = min(3, len(config.years))
    if len(set(selected_years)) < required_year_count:
        raise ValueError(
            f"formula audit requires at least {required_year_count} distinct years"
        )
    if not set(selected_years).issubset(config.years):
        raise ValueError("formula audit years are outside the configured range")
    if max_assets <= 0:
        raise ValueError("max_assets must be positive")
    definitions = {definition.name: definition for definition in PROXY_FACTOR_REGISTRY}
    selected = tuple(definitions[name] for name in FORMULA_AUDIT_FACTORS)
    audit_warmup = max(definition.window or 0 for definition in selected)
    registry_causal = not any(
        ",-" in definition.expression.replace(" ", "")
        for definition in PROXY_FACTOR_REGISTRY
    )
    if not registry_causal:
        return {
            "passed": False,
            "years_checked": list(selected_years),
            "factors_checked": list(FORMULA_AUDIT_FACTORS),
            "comparisons": 0,
            "max_abs_error": None,
            "registry_causal": False,
            "error": "factor registry contains a future Ref expression",
        }
    comparisons = 0
    maximum_error = 0.0
    for year in selected_years:
        start_time = f"{year:04d}-01-01"
        end_time = f"{year:04d}-02-15"
        raw = provider.query(
            [f"${name}" for name in RAW_AUDIT_FIELDS],
            list(RAW_AUDIT_FIELDS),
            start_time,
            end_time,
        )
        actual = provider.query(
            [definition.expression for definition in selected],
            list(FORMULA_AUDIT_FACTORS),
            start_time,
            end_time,
        )
        for frame, expected_columns in (
            (raw, list(RAW_AUDIT_FIELDS)),
            (actual, list(FORMULA_AUDIT_FACTORS)),
        ):
            if not isinstance(frame.index, pd.MultiIndex):
                raise ValueError("formula audit provider result must use a MultiIndex")
            if frame.index.names != ["instrument", "datetime"]:
                raise ValueError(
                    "formula audit provider index must be instrument/datetime"
                )
            if frame.index.has_duplicates or frame.columns.tolist() != expected_columns:
                raise ValueError("formula audit provider returned an invalid schema")
        assets = sorted(
            set(raw.index.get_level_values("instrument"))
            & set(actual.index.get_level_values("instrument"))
        )[:max_assets]
        if not assets:
            raise ValueError(f"formula audit found no shared assets in {year}")
        for asset in assets:
            asset_raw = raw.xs(asset, level="instrument").sort_index()
            expected = compute_proxy_factors_pandas(asset_raw).loc[
                :, FORMULA_AUDIT_FACTORS
            ]
            observed = (
                actual.xs(asset, level="instrument")
                .sort_index()
                .loc[:, FORMULA_AUDIT_FACTORS]
            )
            common_dates = expected.index.intersection(observed.index)
            # Qlib loads pre-start history for Ref and rolling expressions, while
            # the raw audit query cannot. Compare after the largest audit window.
            common_dates = common_dates[audit_warmup:]
            if common_dates.empty:
                raise ValueError(f"formula audit has insufficient dates in {year}")
            for name in FORMULA_AUDIT_FACTORS:
                expected_values = expected.loc[common_dates, name].to_numpy(
                    dtype="float64", copy=False
                )
                observed_values = observed.loc[common_dates, name].to_numpy(
                    dtype="float64", copy=False
                )
                expected_finite = np.isfinite(expected_values)
                observed_finite = np.isfinite(observed_values)
                if not np.array_equal(expected_finite, observed_finite):
                    return {
                        "passed": False,
                        "years_checked": list(selected_years),
                        "factors_checked": list(FORMULA_AUDIT_FACTORS),
                        "comparisons": comparisons,
                        "max_abs_error": maximum_error,
                        "registry_causal": True,
                        "error": (
                            f"{name} finite mask differs from pandas in {year} "
                            f"for {asset}"
                        ),
                    }
                finite = expected_finite
                if not finite.any():
                    return {
                        "passed": False,
                        "years_checked": list(selected_years),
                        "factors_checked": list(FORMULA_AUDIT_FACTORS),
                        "comparisons": comparisons,
                        "max_abs_error": maximum_error,
                        "registry_causal": True,
                        "error": f"{name} has no finite comparisons in {year}",
                    }
                differences = np.abs(expected_values[finite] - observed_values[finite])
                maximum_error = max(maximum_error, float(differences.max()))
                comparisons += int(finite.sum())
                if not np.allclose(
                    expected_values[finite],
                    observed_values[finite],
                    rtol=rtol,
                    atol=atol,
                ):
                    return {
                        "passed": False,
                        "years_checked": list(selected_years),
                        "factors_checked": list(FORMULA_AUDIT_FACTORS),
                        "comparisons": comparisons,
                        "max_abs_error": maximum_error,
                        "registry_causal": True,
                        "error": (
                            f"{name} differs from pandas in {year} for {asset}; "
                            f"max_abs_error={float(differences.max()):.8g}"
                        ),
                    }
    return {
        "passed": True,
        "years_checked": list(selected_years),
        "factors_checked": list(FORMULA_AUDIT_FACTORS),
        "comparisons": comparisons,
        "max_abs_error": maximum_error,
        "registry_causal": True,
    }


def verify_causality_audit(
    config: ProxyFactorConfig,
    provider: ProxyProvider,
) -> dict:
    pandas_result = verify_pandas_causality()
    formula_result = verify_qlib_formula_audit(config, provider)
    return {
        "passed": bool(pandas_result["passed"] and formula_result["passed"]),
        "fingerprint": config.fingerprint,
        "pandas_perturbation": pandas_result,
        "qlib_vs_pandas": formula_result,
    }


def verify_year_boundary(
    config: ProxyFactorConfig,
    provider: ProxyProvider,
    year: int,
    *,
    max_dates: int = 5,
) -> dict:
    if year not in config.years:
        raise ValueError(f"year {year} is outside the configured range")
    long_window = tuple(item for item in PROXY_FACTOR_REGISTRY if item.window == 120)
    names = [item.name for item in long_window]
    factor_path = config.output_root / f"factors/year={year:04d}/part.parquet"
    if not factor_path.is_file():
        raise ValueError(f"missing factor partition: {factor_path}")
    stored = pd.read_parquet(
        factor_path,
        columns=["date", "asset", *names],
    )
    dates = pd.DatetimeIndex(stored["date"].drop_duplicates().sort_values())[:max_dates]
    if dates.empty:
        raise ValueError(f"factor partition has no dates: {factor_path}")
    stored = stored.loc[stored["date"].isin(dates)].sort_values(
        ["date", "asset"],
        kind="stable",
    )
    cross_year = query_factor_range(
        provider,
        start_time=f"{year - 1:04d}-12-01",
        end_time=f"{year:04d}-01-31",
        shard_size=len(long_window),
        registry=long_window,
    )
    expected = normalize_proxy_frame(cross_year, names)
    expected = expected.loc[expected["date"].isin(dates)].sort_values(
        ["date", "asset"],
        kind="stable",
    )
    stored_keys = stored[["date", "asset"]].reset_index(drop=True)
    expected_keys = expected[["date", "asset"]].reset_index(drop=True)
    if not stored_keys.equals(expected_keys):
        return {
            "passed": False,
            "factors_checked": len(names),
            "dates_checked": len(dates),
            "error": "cross-year query keys do not match the stored partition",
        }
    try:
        np.testing.assert_allclose(
            stored[names].to_numpy(dtype="float32", copy=False),
            expected[names].to_numpy(dtype="float32", copy=False),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    except AssertionError as error:
        return {
            "passed": False,
            "factors_checked": len(names),
            "dates_checked": len(dates),
            "error": str(error),
        }
    return {
        "passed": True,
        "factors_checked": len(names),
        "dates_checked": len(dates),
    }


def _atomic_json_write(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_and_validate_state(config: ProxyFactorConfig) -> dict:
    state_path = config.output_root / "_state.json"
    if not state_path.is_file():
        raise ValueError("missing generation state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("fingerprint") != config.fingerprint:
        raise ValueError("state fingerprint does not match configuration")
    expected_years = {str(year) for year in config.years}
    actual_years = set(state.get("years", {}))
    missing = sorted(expected_years - actual_years)
    extra = sorted(actual_years - expected_years)
    if missing:
        raise ValueError(f"missing years in generation state: {missing}")
    if extra:
        raise ValueError(f"unexpected years in generation state: {extra}")
    return state


def _load_listing_intervals(config: ProxyFactorConfig) -> pd.DataFrame:
    path = config.provider_uri / "instruments" / "all.txt"
    if not path.is_file():
        raise ValueError(f"listing interval file is missing: {path}")
    intervals = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["asset", "listing_start", "listing_end"],
        dtype={"asset": str},
        parse_dates=["listing_start", "listing_end"],
    )
    intervals["asset"] = intervals["asset"].str.upper()
    if intervals["asset"].duplicated().any():
        raise ValueError("listing interval file contains duplicate assets")
    return intervals


def _validate_listing_intervals(
    keys: pd.DataFrame,
    intervals: pd.DataFrame,
    year: int,
) -> None:
    merged = keys.assign(asset=keys["asset"].astype(str).str.upper()).merge(
        intervals,
        on="asset",
        how="left",
        validate="many_to_one",
    )
    if merged["listing_start"].isna().any():
        missing = sorted(merged.loc[merged["listing_start"].isna(), "asset"].unique())
        raise ValueError(f"assets are missing from listing intervals: {missing[:5]}")
    dates = pd.to_datetime(merged["date"])
    valid = dates.between(merged["listing_start"], merged["listing_end"])
    if not valid.all():
        row = merged.loc[~valid, ["asset", "date"]].iloc[0]
        raise ValueError(
            f"year {year} row is outside listing interval: "
            f"{row['asset']} {pd.Timestamp(row['date']).date()}"
        )


def _factor_statistics_from_parquet(
    parquet: pq.ParquetFile,
    factor_names: Sequence[str],
) -> dict[str, FactorQualityStats]:
    result: dict[str, FactorQualityStats] = {}
    for name in factor_names:
        column = parquet.read(columns=[name]).column(name)
        values = column.to_pandas().to_numpy(dtype="float32", copy=False)
        result[name] = _factor_stats(values)
    return result


def _validate_physical_schema(
    schema: pa.Schema,
    *,
    value_names: Sequence[str],
    mask_names: Sequence[str] = (),
) -> None:
    if not pa.types.is_timestamp(schema.field("date").type):
        raise ValueError("schema mismatch: date must be a timestamp")
    if not pa.types.is_string(schema.field("asset").type):
        raise ValueError("schema mismatch: asset must be a string")
    for name in value_names:
        if not pa.types.is_float32(schema.field(name).type):
            raise ValueError(f"schema mismatch: {name} must be float32")
    for name in mask_names:
        if not pa.types.is_boolean(schema.field(name).type):
            raise ValueError(f"schema mismatch: {name} must be boolean")


def _validate_year_partition(
    config: ProxyFactorConfig,
    state: dict,
    intervals: pd.DataFrame,
    year: int,
    *,
    include_statistics: bool,
) -> tuple[dict, dict[str, FactorQualityStats], dict]:
    year_state = state.get("years", {}).get(str(year))
    if year_state is None:
        raise ValueError(f"missing year {year} in generation state")
    factor_path = config.output_root / year_state["factor_path"]
    label_path = config.output_root / year_state["label_path"]
    factor_names = [item.name for item in PROXY_FACTOR_REGISTRY]
    label_names = [item.name for item in PROXY_LABEL_REGISTRY]
    expected_factor_schema = ["date", "asset", *factor_names]
    expected_label_schema = [
        "date",
        "asset",
        *label_names,
        *(f"{name}_mask" for name in label_names),
    ]
    parquet_files: dict[Path, pq.ParquetFile] = {}
    for path, checksum_key, expected_schema, row_key, values, masks in (
        (
            factor_path,
            "factor_sha256",
            expected_factor_schema,
            "factor_rows",
            factor_names,
            (),
        ),
        (
            label_path,
            "label_sha256",
            expected_label_schema,
            "label_rows",
            label_names,
            [f"{name}_mask" for name in label_names],
        ),
    ):
        if not path.is_file():
            raise ValueError(f"missing partition: {path}")
        if sha256_file(path) != year_state[checksum_key]:
            raise ValueError(f"checksum mismatch: {path}")
        parquet = pq.ParquetFile(path)
        parquet_files[path] = parquet
        if parquet.schema_arrow.names != expected_schema:
            raise ValueError(f"schema mismatch: {path}")
        _validate_physical_schema(
            parquet.schema_arrow,
            value_names=values,
            mask_names=masks,
        )
        if parquet.metadata.num_rows != int(year_state[row_key]):
            raise ValueError(f"state row count mismatch: {path}")
    if int(year_state.get("factor_columns", -1)) != len(factor_names):
        raise ValueError(f"state factor column count mismatch in year {year}")
    factor_keys = pd.read_parquet(factor_path, columns=["date", "asset"])
    label_frame = pd.read_parquet(label_path)
    label_keys = label_frame[["date", "asset"]]
    duplicate_keys = int(
        factor_keys.duplicated(["date", "asset"]).sum()
        + label_keys.duplicated(["date", "asset"]).sum()
    )
    if duplicate_keys:
        raise ValueError(f"duplicate date/asset keys in year {year}")
    for keys, kind in ((factor_keys, "factor"), (label_keys, "label")):
        dates = pd.to_datetime(keys["date"])
        if not dates.empty and not dates.dt.year.eq(year).all():
            raise ValueError(f"{kind} partition contains dates outside year {year}")
        if not pd.MultiIndex.from_frame(
            keys[["date", "asset"]]
        ).is_monotonic_increasing:
            raise ValueError(f"{kind} partition keys are not sorted")
    if not factor_keys.reset_index(drop=True).equals(label_keys.reset_index(drop=True)):
        raise ValueError(f"factor/label keys do not match in year {year}")
    _validate_listing_intervals(factor_keys, intervals, year)
    for name in label_names:
        values = label_frame[name].to_numpy(dtype="float32", copy=False)
        if np.isinf(values).any():
            raise ValueError(f"label {name} contains nonfinite values in year {year}")
        expected_mask = np.isfinite(values)
        actual_mask = label_frame[f"{name}_mask"].to_numpy(dtype=np.bool_, copy=False)
        if not np.array_equal(expected_mask, actual_mask):
            raise ValueError(f"label mask mismatch for {name} in year {year}")
    nonfinite_values = 0
    factor_parquet = parquet_files[factor_path]
    for batch in factor_parquet.iter_batches(
        batch_size=65_536,
        columns=factor_names,
    ):
        for column in batch.columns:
            values = column.to_pandas().to_numpy(dtype="float32", copy=False)
            nonfinite_values += int(np.isinf(values).sum())
    if nonfinite_values:
        raise ValueError(f"year {year} contains {nonfinite_values} nonfinite values")
    statistics = (
        _factor_statistics_from_parquet(factor_parquet, factor_names)
        if include_statistics
        else {}
    )
    report = {
        "year": year,
        "factor_rows": len(factor_keys),
        "label_rows": len(label_keys),
        "factor_columns": len(factor_names),
        "duplicate_keys": duplicate_keys,
        "nonfinite_values": nonfinite_values,
        "checksums_valid": True,
        "schemas_valid": True,
        "keys_aligned": True,
        "masks_valid": True,
        "listing_intervals_valid": True,
    }
    partition_record = {
        **year_state,
        "factor_bytes": factor_path.stat().st_size,
        "label_bytes": label_path.stat().st_size,
    }
    return report, statistics, partition_record


def verify_materialized_year(config: ProxyFactorConfig, year: int) -> dict:
    if year not in config.years:
        raise ValueError(f"year {year} is outside the configured range")
    state_path = config.output_root / "_state.json"
    if not state_path.is_file():
        raise ValueError("missing generation state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("fingerprint") != config.fingerprint:
        raise ValueError("state fingerprint does not match configuration")
    report, _, _ = _validate_year_partition(
        config,
        state,
        _load_listing_intervals(config),
        year,
        include_statistics=False,
    )
    return report


def _validate_partition_files(
    config: ProxyFactorConfig,
    state: dict,
) -> tuple[int, dict[str, dict], dict[str, dict]]:
    intervals = _load_listing_intervals(config)
    total_rows = 0
    yearly_statistics: dict[str, dict] = {}
    partition_records: dict[str, dict] = {}
    for year in config.years:
        report, statistics, record = _validate_year_partition(
            config,
            state,
            intervals,
            year,
            include_statistics=True,
        )
        total_rows += report["factor_rows"]
        yearly_statistics[str(year)] = {
            name: asdict(item) for name, item in statistics.items()
        }
        partition_records[str(year)] = record
    return total_rows, yearly_statistics, partition_records


def _global_factor_statistics(
    config: ProxyFactorConfig,
) -> dict[str, FactorQualityStats]:
    dataset = ds.dataset(
        config.output_root / "factors",
        format="parquet",
        partitioning="hive",
    )
    result: dict[str, FactorQualityStats] = {}
    for definition in PROXY_FACTOR_REGISTRY:
        column = dataset.to_table(columns=[definition.name]).column(definition.name)
        values = column.to_pandas().to_numpy(dtype="float32", copy=False)
        result[definition.name] = _factor_stats(values)
    return result


def _validate_causality_audit(
    config: ProxyFactorConfig,
    audit: dict | None,
) -> dict:
    if not audit or not audit.get("passed"):
        raise ValueError("a passing causality audit is required before finalization")
    if audit.get("fingerprint") != config.fingerprint:
        raise ValueError("causality audit fingerprint does not match configuration")
    pandas_result = audit.get("pandas_perturbation", {})
    formula_result = audit.get("qlib_vs_pandas", {})
    if not pandas_result.get("passed") or pandas_result.get("factors_checked") != 128:
        raise ValueError("causality audit pandas perturbation evidence is incomplete")
    years = tuple(formula_result.get("years_checked", []))
    factors = tuple(formula_result.get("factors_checked", []))
    if (
        not formula_result.get("passed")
        or not formula_result.get("registry_causal")
        or years != _default_audit_years(config)
        or factors != FORMULA_AUDIT_FACTORS
        or int(formula_result.get("comparisons", 0)) <= 0
    ):
        raise ValueError("causality audit Qlib/pandas evidence is incomplete")
    return audit


def finalize_dataset(
    config: ProxyFactorConfig,
    *,
    completed_years: Iterable[int] | None = None,
    causality_audit: dict | None = None,
) -> dict:
    manifest_path = config.output_root / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    if completed_years is not None:
        supplied = set(int(year) for year in completed_years)
        expected = set(config.years)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing:
            raise ValueError(f"missing years: {missing}")
        if extra:
            raise ValueError(f"unexpected years: {extra}")
    state = _load_and_validate_state(config)
    total_rows, yearly_statistics, partition_records = _validate_partition_files(
        config,
        state,
    )
    stats = _global_factor_statistics(config)
    validated_audit = _validate_causality_audit(config, causality_audit)
    nonfinite_values = sum(item.nonfinite_count for item in stats.values())
    failures: list[str] = []
    if nonfinite_values:
        failures.append(f"{nonfinite_values} nonfinite factor values")
    for name, item in stats.items():
        if item.valid_ratio < config.min_global_valid_ratio:
            failures.append(
                f"{name} valid_ratio {item.valid_ratio:.6f} is below "
                f"{config.min_global_valid_ratio:.6f}"
            )
        if item.near_constant_ratio > config.max_near_constant_ratio:
            failures.append(
                f"{name} near_constant_ratio {item.near_constant_ratio:.6f} is above "
                f"{config.max_near_constant_ratio:.6f}"
            )
    if failures:
        raise ValueError("quality gates failed: " + "; ".join(failures))
    quality_report = {
        "version": 1,
        "fingerprint": config.fingerprint,
        "years": list(config.years),
        "total_rows": total_rows,
        "duplicate_keys": 0,
        "nonfinite_values": nonfinite_values,
        "yearly_factors": yearly_statistics,
        "factors": {name: asdict(item) for name, item in stats.items()},
        "causality_audit": validated_audit,
    }
    quality_path = config.output_root / "quality_report.json"
    _atomic_json_write(quality_report, quality_path)
    try:
        qlib_version = metadata.version("pyqlib")
    except metadata.PackageNotFoundError:
        qlib_version = "unknown"
    manifest = {
        "version": 1,
        "dataset": "ProxyFactor-v0",
        "complete": True,
        "fingerprint": config.fingerprint,
        "provider_uri": str(config.provider_uri),
        "qlib_version": qlib_version,
        "universe": config.universe,
        "frequency": config.frequency,
        "years": list(config.years),
        "factors": [asdict(item) for item in PROXY_FACTOR_REGISTRY],
        "labels": [asdict(item) for item in PROXY_LABEL_REGISTRY],
        "partitions": partition_records,
        "quality_report": quality_path.name,
        "quality_report_sha256": sha256_file(quality_path),
    }
    _atomic_json_write(manifest, manifest_path)
    return manifest
