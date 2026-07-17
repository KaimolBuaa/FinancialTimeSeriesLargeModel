"""Recoverable manifest-driven archive download batches."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import HTTPError

from .binance_um import archive_url, download_verified_archive
from .config import PipelineConfig, iter_months


DownloadFunction = Callable[..., Path]


def _manifest_path(data_root: Path) -> Path:
    return data_root / "manifests" / "raw_downloads.jsonl"


def _terminal_items(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    terminal: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (record["symbol"], record["month"])
        if record["status"] in {"downloaded", "missing"}:
            terminal.add(key)
        elif record["status"] == "failed":
            terminal.discard(key)
    return terminal


def _append_record(path: Path, record: dict[str, str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def run_downloads(
    config: PipelineConfig,
    *,
    downloader: DownloadFunction = download_verified_archive,
    max_items: int | None = None,
) -> list[dict[str, str | int]]:
    """Download all nonterminal symbol-months and append durable manifest records."""

    manifest = _manifest_path(config.data_root)
    terminal = _terminal_items(manifest)
    records: list[dict[str, str | int]] = []
    for month in iter_months(config):
        for symbol in config.symbols:
            if max_items is not None and len(records) >= max_items:
                return records
            if (symbol, month) in terminal:
                continue
            record: dict[str, str | int] = {
                "symbol": symbol,
                "month": month,
                "source_url": archive_url(symbol, month),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                path = downloader(
                    data_root=config.data_root,
                    symbol=symbol,
                    month=month,
                    target_bytes=config.target_bytes,
                    hard_bytes=config.hard_bytes,
                )
                record.update({"status": "downloaded", "path": str(path)})
            except HTTPError as error:
                if error.code == 404:
                    record.update({"status": "missing", "http_status": error.code})
                else:
                    record.update(
                        {
                            "status": "failed",
                            "http_status": error.code,
                            "error": str(error),
                        }
                    )
            except Exception as error:  # noqa: BLE001 - failures belong in the manifest.
                record.update({"status": "failed", "error": str(error)})
            _append_record(manifest, record)
            records.append(record)
    return records
