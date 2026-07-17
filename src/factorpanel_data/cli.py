"""Command-line entry points for resumable Binance UM data preparation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .batch import run_downloads
from .config import PipelineConfig, iter_months, load_config


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


def render_progress_bar(*, completed: int, total: int, width: int = 30) -> str:
    """Render a fixed-width, bounded terminal progress bar."""

    if total <= 0:
        raise ValueError("total must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    bounded = min(max(completed, 0), total)
    filled = int(bounded / total * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {bounded / total:.1%} ({bounded}/{total})"


def progress_summary(config: PipelineConfig) -> dict[str, int]:
    """Report terminal work, pending work, and errors for the configured universe."""

    summary = manifest_summary(config.data_root)
    total = len(config.symbols) * len(iter_months(config))
    completed = summary["downloaded"] + summary["missing"]
    return {**summary, "total": total, "completed": completed, "pending": total - completed}


def format_progress(summary: dict[str, int]) -> str:
    """Format a concise, human-readable download status line."""

    return (
        f"{render_progress_bar(completed=summary['completed'], total=summary['total'])} "
        f"pending={summary['pending']} downloaded={summary['downloaded']} "
        f"missing={summary['missing']} failed={summary['failed']}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subcommands = parser.add_subparsers(dest="command", required=True)
    download = subcommands.add_parser("download", help="download pending verified raw 1m archives")
    download.add_argument("--max-items", type=int)
    status = subcommands.add_parser("status", help="report latest raw-download manifest state")
    status.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a download batch or report its durable manifest state."""

    args = _parser().parse_args(argv)
    config = load_config(args.config)
    summary = progress_summary(config)
    if args.command == "download":
        records = run_downloads(config, max_items=args.max_items)
        print(json.dumps({"processed": len(records), **progress_summary(config)}, sort_keys=True))
        return 0
    if args.as_json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(format_progress(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
