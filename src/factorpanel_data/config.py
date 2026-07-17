"""Configuration and deterministic month planning for the crypto data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


GIB = 1024**3


@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path
    symbols: tuple[str, ...]
    start_month: str
    end_month: str
    target_bytes: int
    hard_bytes: int


def _month(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m")


def iter_months(config: PipelineConfig) -> list[str]:
    """Return the inclusive month range in stable chronological order."""

    start = _month(config.start_month)
    end = _month(config.end_month)
    if start > end:
        raise ValueError("start_month must not be after end_month")
    months: list[str] = []
    cursor = start
    while cursor <= end:
        months.append(cursor.strftime("%Y-%m"))
        cursor = datetime(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    return months


def load_config(path: Path) -> PipelineConfig:
    """Load a JSON configuration and reject unsafe storage budgets."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    config = PipelineConfig(
        data_root=Path(raw["data_root"]),
        symbols=tuple(raw["symbols"]),
        start_month=str(raw["start_month"]),
        end_month=str(raw["end_month"]),
        target_bytes=int(raw.get("target_gib", 60) * GIB),
        hard_bytes=int(raw.get("hard_gib", 80) * GIB),
    )
    if not config.symbols:
        raise ValueError("symbols must not be empty")
    if config.target_bytes > config.hard_bytes:
        raise ValueError("target_gib must not exceed hard_gib")
    iter_months(config)
    return config
