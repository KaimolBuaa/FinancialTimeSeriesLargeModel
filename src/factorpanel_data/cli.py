"""Command-line entry points for resumable Binance UM data preparation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .batch import run_downloads
from .config import load_config


def manifest_summary(data_root: Path) -> dict[str, int]:
    """Count the latest state of every symbol-month recorded in the manifest."""

    manifest = data_root / "manifests" / "raw_downloads.jsonl"
    latest: dict[tuple[str, str], str] = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            latest[(record["symbol"], record["month"])] = record["status"]
    counts = Counter(latest.values())
    return {status: counts.get(status, 0) for status in ("downloaded", "missing", "failed")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subcommands = parser.add_subparsers(dest="command", required=True)
    download = subcommands.add_parser("download", help="download pending verified raw 1m archives")
    download.add_argument("--max-items", type=int)
    subcommands.add_parser("status", help="report latest raw-download manifest state")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a download batch or report its durable manifest state."""

    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "download":
        records = run_downloads(config, max_items=args.max_items)
        print(json.dumps({"processed": len(records), **manifest_summary(config.data_root)}, sort_keys=True))
        return 0
    print(json.dumps(manifest_summary(config.data_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
