"""Lazy Qlib access and deterministic ProxyFactor-v0 queries."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

from .proxy_registry import (
    PROXY_FACTOR_REGISTRY,
    PROXY_LABEL_REGISTRY,
    FactorDefinition,
    LabelDefinition,
)


class ProxyProvider(Protocol):
    def query(
        self,
        fields: Sequence[str],
        names: Sequence[str],
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame: ...


class QlibProxyProvider:
    def __init__(
        self,
        provider_uri: Path | str,
        universe: str = "all",
        *,
        kernels: int = 1,
    ) -> None:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D

        qlib.init(
            provider_uri=str(Path(provider_uri).resolve()),
            region=REG_CN,
            kernels=kernels,
        )
        self._features = D.features
        self._instruments = D.instruments(universe)

    def query(
        self,
        fields: Sequence[str],
        names: Sequence[str],
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        frame = self._features(
            self._instruments,
            fields=list(fields),
            start_time=start_time,
            end_time=end_time,
            freq="day",
            disk_cache=0,
        )
        if frame.shape[1] != len(names):
            raise ValueError("Qlib returned an unexpected number of columns")
        frame.columns = list(names)
        return frame


def _validate_frame(frame: pd.DataFrame, names: Sequence[str]) -> None:
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("provider result must use a MultiIndex")
    if frame.index.names != ["instrument", "datetime"]:
        raise ValueError("provider index must be named instrument/datetime")
    if frame.index.has_duplicates:
        raise ValueError("provider returned a duplicate index")
    if frame.columns.has_duplicates:
        raise ValueError("provider returned duplicate columns")
    if frame.columns.tolist() != list(names):
        raise ValueError("provider returned an unexpected schema")


def _trim_dates(frame: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame.index.get_level_values("datetime"))
    keep = (dates >= pd.Timestamp(start_time)) & (dates <= pd.Timestamp(end_time))
    return frame.loc[keep].sort_index()


def query_factor_range(
    provider: ProxyProvider,
    *,
    start_time: str,
    end_time: str,
    shard_size: int = 32,
    registry: Sequence[FactorDefinition] = PROXY_FACTOR_REGISTRY,
) -> pd.DataFrame:
    if shard_size <= 0 or len(registry) % shard_size != 0:
        raise ValueError("shard_size must divide the factor registry")
    parts: list[pd.DataFrame] = []
    for offset in range(0, len(registry), shard_size):
        shard = registry[offset : offset + shard_size]
        names = [item.name for item in shard]
        frame = provider.query(
            [item.expression for item in shard],
            names,
            start_time,
            end_time,
        )
        _validate_frame(frame, names)
        parts.append(frame)
    result = pd.concat(parts, axis=1, join="outer")
    if result.columns.has_duplicates:
        raise ValueError("factor shards produced duplicate columns")
    expected_names = [item.name for item in registry]
    if result.columns.tolist() != expected_names:
        raise ValueError("factor shards produced an unexpected schema")
    if result.index.has_duplicates:
        raise ValueError("factor shards produced a duplicate index")
    return _trim_dates(result, start_time, end_time)


def query_factor_year(
    provider: ProxyProvider,
    *,
    year: int,
    shard_size: int = 32,
    registry: Sequence[FactorDefinition] = PROXY_FACTOR_REGISTRY,
) -> pd.DataFrame:
    return query_factor_range(
        provider,
        start_time=f"{year:04d}-01-01",
        end_time=f"{year:04d}-12-31",
        shard_size=shard_size,
        registry=registry,
    )


def query_label_year(
    provider: ProxyProvider,
    *,
    year: int,
    registry: Sequence[LabelDefinition] = PROXY_LABEL_REGISTRY,
) -> pd.DataFrame:
    names = [item.name for item in registry]
    frame = provider.query(
        [item.expression for item in registry],
        names,
        f"{year:04d}-01-01",
        f"{year + 1:04d}-02-15",
    )
    _validate_frame(frame, names)
    return _trim_dates(frame, f"{year:04d}-01-01", f"{year:04d}-12-31")
