"""Operational CLI for ProxyFactor-v0 generation and verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Sequence
import uuid

import numpy as np

from .proxy_config import ProxyFactorConfig, load_proxy_config
from .proxy_materialize import materialize_year
from .proxy_quality import (
    finalize_dataset,
    verify_causality_audit,
    verify_materialized_year,
    verify_year_boundary,
)
from .proxy_store import ProxyFactorStore
from .qlib_proxy import QlibProxyProvider, query_factor_range


def dataset_status(config: ProxyFactorConfig) -> dict:
    state_path = config.output_root / "_state.json"
    state = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state_years = state.get("years", {}) if isinstance(state, dict) else {}
    completed_years: list[int] = []
    missing_factor_years: list[int] = []
    missing_label_years: list[int] = []
    for year in config.years:
        factor_path = config.output_root / f"factors/year={year:04d}/part.parquet"
        label_path = config.output_root / f"labels/year={year:04d}/part.parquet"
        factor_exists = factor_path.is_file()
        label_exists = label_path.is_file()
        if not factor_exists:
            missing_factor_years.append(year)
        if not label_exists:
            missing_label_years.append(year)
        if factor_exists and label_exists and str(year) in state_years:
            completed_years.append(year)
    manifest_path = config.output_root / "manifest.json"
    complete = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = bool(
            manifest.get("complete") is True
            and manifest.get("fingerprint") == config.fingerprint
            and completed_years == list(config.years)
        )
    return {
        "complete": complete,
        "fingerprint": config.fingerprint,
        "state_fingerprint_matches": (
            not state or state.get("fingerprint") == config.fingerprint
        ),
        "completed_years": completed_years,
        "missing_factor_years": missing_factor_years,
        "missing_label_years": missing_label_years,
    }


def _preflight_generation(
    config: ProxyFactorConfig,
    *,
    force_years: set[int],
) -> bool:
    state_path = config.output_root / "_state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("fingerprint") == config.fingerprint:
        return False
    existing_years = {int(year) for year in state.get("years", {})}
    if not existing_years or existing_years.issubset(force_years):
        return True
    raise ValueError(
        "generation fingerprint does not match existing state; "
        "force every existing year to replace it"
    )


def prepare_for_forced_migration(
    config: ProxyFactorConfig,
    force_years: set[int],
) -> Path | None:
    state_path = config.output_root / "_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("fingerprint") == config.fingerprint:
        return None
    existing_years = {int(year) for year in state.get("years", {})}
    if not existing_years.issubset(force_years):
        raise ValueError(
            "generation fingerprint does not match existing state; "
            "force every existing year to replace it"
        )
    backup = config.output_root / (
        f"_state.{state.get('fingerprint', 'unknown')}.{uuid.uuid4().hex}.json"
    )
    shutil.copy2(state_path, backup)
    temporary = state_path.with_name(f".{state_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "fingerprint": config.fingerprint,
                    "years": {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, state_path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def _emit(payload: dict, *, stream=sys.stdout) -> None:
    print(json.dumps(payload, sort_keys=True), file=stream, flush=True)


def _generate(args: argparse.Namespace, config: ProxyFactorConfig) -> dict:
    start_year = args.start_year if args.start_year is not None else config.start_year
    end_year = args.end_year if args.end_year is not None else config.end_year
    if start_year > end_year:
        raise ValueError("start-year must not be after end-year")
    years = tuple(range(start_year, end_year + 1))
    if not set(years).issubset(config.years):
        raise ValueError("requested generation years are outside the configuration")
    force_years = set(args.force_year or [])
    migration_required = _preflight_generation(
        config,
        force_years=force_years,
    )
    provider = QlibProxyProvider(
        config.provider_uri,
        universe=config.universe,
        kernels=args.kernels,
    )
    if migration_required:
        prepare_for_forced_migration(config, force_years)
    generated = []
    skipped = []
    for year in years:
        _emit({"event": "year_start", "year": year}, stream=sys.stderr)
        result = materialize_year(
            config,
            provider,
            year,
            resume=args.resume and year not in force_years,
            force=year in force_years,
        )
        (skipped if result.skipped else generated).append(year)
        _emit(
            {
                "event": "year_complete",
                "year": year,
                "rows": result.factor_rows,
                "skipped": result.skipped,
            },
            stream=sys.stderr,
        )
    return {
        "generated_years": generated,
        "skipped_years": skipped,
        "status": dataset_status(config),
    }


def _verify(args: argparse.Namespace, config: ProxyFactorConfig) -> dict:
    years = tuple(args.year) if args.year else config.years
    reports = {str(year): verify_materialized_year(config, year) for year in years}
    payload: dict = {"years": reports}
    provider = None
    causality = None
    if args.causality or args.boundary or args.finalize:
        provider = QlibProxyProvider(
            config.provider_uri,
            universe=config.universe,
            kernels=args.kernels,
        )
    if args.causality or args.finalize:
        assert provider is not None
        causality = verify_causality_audit(config, provider)
        if not causality["passed"]:
            formula_error = causality["qlib_vs_pandas"].get("error", "unknown")
            raise ValueError(f"causality verification failed: {formula_error}")
        payload["causality"] = causality
    if args.boundary:
        assert provider is not None
        boundary = {
            str(year): verify_year_boundary(config, provider, year) for year in years
        }
        failures = [year for year, report in boundary.items() if not report["passed"]]
        if failures:
            raise ValueError(f"year-boundary verification failed: {failures}")
        payload["boundary"] = boundary
    if args.finalize:
        payload["manifest"] = finalize_dataset(
            config,
            causality_audit=causality,
        )
    payload["status"] = dataset_status(config)
    return payload


def _sample(args: argparse.Namespace, config: ProxyFactorConfig) -> dict:
    panel = ProxyFactorStore(config.output_root).read_panel(
        factor=args.factor,
        end_date=args.end_date,
        context_length=args.context_length,
        max_assets=args.max_assets,
        seed=args.seed,
    )
    return {
        "factor": panel.factor,
        "shape": list(panel.values.shape),
        "observed_ratio": float(panel.observed_mask.mean()),
        "finite_values": bool(np.isfinite(panel.values).all()),
        "date_start": str(panel.dates[0]),
        "date_end": str(panel.dates[-1]),
        "assets": len(panel.assets),
    }


def _sample_query(args: argparse.Namespace, config: ProxyFactorConfig) -> dict:
    provider = QlibProxyProvider(
        config.provider_uri,
        universe=config.universe,
        kernels=args.kernels,
    )
    frame = query_factor_range(
        provider,
        start_time=args.start,
        end_time=args.end,
        shard_size=config.factor_shard_size,
    )
    values = frame.to_numpy(dtype="float64", copy=False)
    dates = frame.index.get_level_values("datetime")
    return {
        "rows": len(frame),
        "factor_columns": frame.shape[1],
        "assets": int(frame.index.get_level_values("instrument").nunique()),
        "date_start": str(dates.min().date()) if len(dates) else None,
        "date_end": str(dates.max().date()) if len(dates) else None,
        "duplicate_keys": int(frame.index.duplicated().sum()),
        "infinite_values": int(np.isinf(values).sum()),
        "valid_ratio": float(np.isfinite(values).sum() / values.size)
        if values.size
        else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="factorpanel-proxy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--start-year", type=int)
    generate.add_argument("--end-year", type=int)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--force-year", type=int, action="append")
    generate.add_argument("--kernels", type=int, default=4)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", required=True)
    verify.add_argument("--year", type=int, action="append")
    verify.add_argument("--causality", action="store_true")
    verify.add_argument("--boundary", action="store_true")
    verify.add_argument("--finalize", action="store_true")
    verify.add_argument("--kernels", type=int, default=4)

    status = subparsers.add_parser("status")
    status.add_argument("--config", required=True)
    status.add_argument("--json", action="store_true")

    sample = subparsers.add_parser("sample")
    sample.add_argument("--config", required=True)
    sample.add_argument("--factor", required=True)
    sample.add_argument("--end-date", required=True)
    sample.add_argument("--context-length", type=int, default=256)
    sample.add_argument("--max-assets", type=int, default=512)
    sample.add_argument("--seed", type=int, default=0)

    sample_query = subparsers.add_parser("sample-query")
    sample_query.add_argument("--config", required=True)
    sample_query.add_argument("--start", required=True)
    sample_query.add_argument("--end", required=True)
    sample_query.add_argument("--kernels", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_proxy_config(Path(args.config))
        if args.command == "generate":
            payload = _generate(args, config)
        elif args.command == "verify":
            payload = _verify(args, config)
        elif args.command == "status":
            payload = dataset_status(config)
        elif args.command == "sample":
            payload = _sample(args, config)
        elif args.command == "sample-query":
            payload = _sample_query(args, config)
        else:
            raise ValueError(f"unsupported command: {args.command}")
    except Exception as error:
        _emit({"ok": False, "error": str(error)}, stream=sys.stderr)
        return 1
    _emit({"ok": True, **payload})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
