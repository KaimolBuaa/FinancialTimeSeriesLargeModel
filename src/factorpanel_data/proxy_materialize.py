"""Atomic yearly materialization and resume state for ProxyFactor-v0."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Sequence
import uuid

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .proxy_config import ProxyFactorConfig
from .proxy_registry import PROXY_FACTOR_REGISTRY, PROXY_LABEL_REGISTRY
from .qlib_proxy import ProxyProvider, query_factor_year, query_label_year


ParquetWriter = Callable[[pd.DataFrame, Path, str], None]


@dataclass(frozen=True)
class MaterializeResult:
    year: int
    factor_path: Path
    label_path: Path
    factor_rows: int
    label_rows: int
    factor_columns: int
    factor_sha256: str
    label_sha256: str
    skipped: bool = False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_proxy_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("proxy frame must use a two-level MultiIndex")
    ordered = frame.loc[:, list(columns)].replace([np.inf, -np.inf], np.nan)
    ordered = ordered.astype("float32")
    ordered.index = ordered.index.set_names(["asset", "date"])
    result = ordered.reset_index()
    result["date"] = pd.to_datetime(result["date"])
    result["asset"] = result["asset"].astype(str)
    result = result.sort_values(["date", "asset"], kind="stable").reset_index(
        drop=True
    )
    result = result.loc[:, ["date", "asset", *columns]]
    if result.duplicated(["date", "asset"]).any():
        raise ValueError("proxy partition contains duplicate date/asset keys")
    return result


def normalize_label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    names = [item.name for item in PROXY_LABEL_REGISTRY]
    result = normalize_proxy_frame(frame, names)
    for name in names:
        result[f"{name}_mask"] = result[name].notna()
    return result


def _default_parquet_writer(
    frame: pd.DataFrame,
    destination: Path,
    compression: str,
) -> None:
    frame.to_parquet(destination, index=False, compression=compression)


def atomic_parquet_write(
    frame: pd.DataFrame,
    destination: Path,
    compression: str = "zstd",
    *,
    writer: ParquetWriter = _default_parquet_writer,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        writer(frame, temporary, compression)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _partition_paths(config: ProxyFactorConfig, year: int) -> tuple[Path, Path]:
    factor_path = config.output_root / f"factors/year={year:04d}/part.parquet"
    label_path = config.output_root / f"labels/year={year:04d}/part.parquet"
    return factor_path, label_path


def _load_state(config: ProxyFactorConfig) -> dict:
    state_path = config.output_root / "_state.json"
    if not state_path.exists():
        return {"version": 1, "fingerprint": config.fingerprint, "years": {}}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid resume state: {state_path}") from error
    if not isinstance(state.get("years"), dict):
        raise ValueError("invalid resume state: years must be an object")
    return state


def _write_state(config: ProxyFactorConfig, state: dict) -> None:
    destination = config.output_root / "_state.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_parquet(
    path: Path,
    expected_columns: Sequence[str],
    expected_rows: int,
) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.names != list(expected_columns):
        raise ValueError(f"unexpected Parquet schema: {path}")
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"unexpected Parquet row count: {path}")


def _publish_pair(
    factor_temporary: Path,
    factor_path: Path,
    label_temporary: Path,
    label_path: Path,
) -> None:
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for destination in (factor_path, label_path):
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.backup"
                )
                os.replace(destination, backup)
                backups[destination] = backup
        for temporary, destination in (
            (factor_temporary, factor_path),
            (label_temporary, label_path),
        ):
            os.replace(temporary, destination)
            published.append(destination)
    except Exception:
        for destination in published:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _resume_result(
    config: ProxyFactorConfig,
    year: int,
    state: dict,
) -> MaterializeResult:
    if state.get("fingerprint") != config.fingerprint:
        raise ValueError("resume fingerprint does not match the current configuration")
    year_state = state["years"].get(str(year))
    if year_state is None:
        raise ValueError(f"resume state does not contain year {year}")
    factor_path, label_path = _partition_paths(config, year)
    for path, key in (
        (factor_path, "factor_sha256"),
        (label_path, "label_sha256"),
    ):
        if not path.is_file():
            raise ValueError(f"resume partition is missing: {path}")
        if sha256_file(path) != year_state.get(key):
            raise ValueError(f"resume checksum mismatch: {path}")
    return MaterializeResult(
        year=year,
        factor_path=factor_path,
        label_path=label_path,
        factor_rows=int(year_state["factor_rows"]),
        label_rows=int(year_state["label_rows"]),
        factor_columns=int(year_state["factor_columns"]),
        factor_sha256=str(year_state["factor_sha256"]),
        label_sha256=str(year_state["label_sha256"]),
        skipped=True,
    )


def materialize_year(
    config: ProxyFactorConfig,
    provider: ProxyProvider,
    year: int,
    *,
    resume: bool = False,
    force: bool = False,
    writer: ParquetWriter = _default_parquet_writer,
) -> MaterializeResult:
    if year not in config.years:
        raise ValueError(f"year {year} is outside the configured range")
    factor_path, label_path = _partition_paths(config, year)
    state = _load_state(config)
    if resume:
        if state.get("fingerprint") != config.fingerprint:
            raise ValueError(
                "resume fingerprint does not match the current configuration"
            )
        if str(year) in state.get("years", {}):
            return _resume_result(config, year, state)
        if factor_path.exists() or label_path.exists():
            raise ValueError(
                "resume partition exists without matching state; use force"
            )
    if not force and (factor_path.exists() or label_path.exists()):
        raise ValueError("partition already exists; use resume or force")
    if state.get("fingerprint") != config.fingerprint:
        other_years = set(state.get("years", {})) - {str(year)}
        if other_years:
            raise ValueError("fingerprint mismatch would mix generation contracts")
        state = {"version": 1, "fingerprint": config.fingerprint, "years": {}}

    factor_names = [item.name for item in PROXY_FACTOR_REGISTRY]
    factor_frame = normalize_proxy_frame(
        query_factor_year(
            provider,
            year=year,
            shard_size=config.factor_shard_size,
        ),
        factor_names,
    )
    label_frame = normalize_label_frame(query_label_year(provider, year=year))
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    factor_temporary = factor_path.with_name(
        f".{factor_path.name}.{uuid.uuid4().hex}.tmp"
    )
    label_temporary = label_path.with_name(
        f".{label_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        writer(factor_frame, factor_temporary, config.compression)
        writer(label_frame, label_temporary, config.compression)
        _validate_parquet(
            factor_temporary,
            ["date", "asset", *factor_names],
            len(factor_frame),
        )
        label_names = [item.name for item in PROXY_LABEL_REGISTRY]
        _validate_parquet(
            label_temporary,
            [
                "date",
                "asset",
                *label_names,
                *(f"{name}_mask" for name in label_names),
            ],
            len(label_frame),
        )
        _publish_pair(
            factor_temporary,
            factor_path,
            label_temporary,
            label_path,
        )
    finally:
        factor_temporary.unlink(missing_ok=True)
        label_temporary.unlink(missing_ok=True)

    factor_sha256 = sha256_file(factor_path)
    label_sha256 = sha256_file(label_path)
    year_state = {
        "factor_path": str(factor_path.relative_to(config.output_root)),
        "label_path": str(label_path.relative_to(config.output_root)),
        "factor_rows": len(factor_frame),
        "label_rows": len(label_frame),
        "factor_columns": len(factor_names),
        "factor_sha256": factor_sha256,
        "label_sha256": label_sha256,
    }
    state["version"] = 1
    state["fingerprint"] = config.fingerprint
    state.setdefault("years", {})[str(year)] = year_state
    _write_state(config, state)
    return MaterializeResult(
        year=year,
        factor_path=factor_path,
        label_path=label_path,
        factor_rows=len(factor_frame),
        label_rows=len(label_frame),
        factor_columns=len(factor_names),
        factor_sha256=factor_sha256,
        label_sha256=label_sha256,
    )
