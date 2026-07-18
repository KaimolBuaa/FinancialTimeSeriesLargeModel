"""Validated configuration for ProxyFactor-v0 generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .proxy_registry import PROXY_FACTOR_REGISTRY, PROXY_LABEL_REGISTRY


LOCAL_CALENDAR_START_YEAR = 2000
LOCAL_CALENDAR_END_YEAR = 2026


@dataclass(frozen=True)
class ProxyFactorConfig:
    provider_uri: Path
    output_root: Path
    start_year: int = 2008
    end_year: int = 2025
    universe: str = "all"
    frequency: str = "day"
    warmup_trading_days: int = 120
    factor_shard_size: int = 32
    compression: str = "zstd"
    min_global_valid_ratio: float = 0.05
    max_near_constant_ratio: float = 0.99
    project_root: Path | None = None

    def __post_init__(self) -> None:
        provider_uri = Path(self.provider_uri).expanduser().resolve()
        output_root = Path(self.output_root).expanduser().resolve()
        project_root = (
            Path(self.project_root).expanduser().resolve()
            if self.project_root is not None
            else None
        )
        object.__setattr__(self, "provider_uri", provider_uri)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "project_root", project_root)

        if not provider_uri.is_dir():
            raise ValueError(f"Qlib provider path does not exist: {provider_uri}")
        if self.start_year > self.end_year:
            raise ValueError("start_year must not be after end_year")
        if self.start_year < LOCAL_CALENDAR_START_YEAR:
            raise ValueError("start_year is outside the local Qlib calendar")
        if self.end_year > LOCAL_CALENDAR_END_YEAR:
            raise ValueError("end_year is outside the local Qlib calendar")
        if self.universe != "all":
            raise ValueError("ProxyFactor-v0 universe must be 'all'")
        if self.frequency != "day":
            raise ValueError("ProxyFactor-v0 frequency must be 'day'")
        if self.factor_shard_size <= 0 or 128 % self.factor_shard_size != 0:
            raise ValueError("factor_shard_size must divide 128")
        if self.warmup_trading_days < max(
            item.window or 0 for item in PROXY_FACTOR_REGISTRY
        ):
            raise ValueError("warmup_trading_days must cover the longest factor window")
        if not 0.0 < self.min_global_valid_ratio <= 1.0:
            raise ValueError("min_global_valid_ratio must be in (0, 1]")
        if not 0.0 <= self.max_near_constant_ratio <= 1.0:
            raise ValueError("max_near_constant_ratio must be in [0, 1]")
        if project_root is not None:
            self._validate_output_location(project_root)

    def _validate_output_location(self, project_root: Path) -> None:
        protected = tuple(
            (project_root / name).resolve() for name in ("src", "tests", "configs")
        )
        for path in protected:
            if self.output_root == path or path in self.output_root.parents:
                raise ValueError(f"output_root overlaps protected project path: {path}")
            if self.output_root in path.parents:
                raise ValueError("output_root must not contain protected project paths")

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(range(self.start_year, self.end_year + 1))

    @property
    def fingerprint(self) -> str:
        payload = {
            "config": {
                "provider_uri": str(self.provider_uri),
                "output_root": str(self.output_root),
                "start_year": self.start_year,
                "end_year": self.end_year,
                "universe": self.universe,
                "frequency": self.frequency,
                "warmup_trading_days": self.warmup_trading_days,
                "factor_shard_size": self.factor_shard_size,
                "compression": self.compression,
                "min_global_valid_ratio": self.min_global_valid_ratio,
                "max_near_constant_ratio": self.max_near_constant_ratio,
            },
            "factors": [asdict(item) for item in PROXY_FACTOR_REGISTRY],
            "labels": [asdict(item) for item in PROXY_LABEL_REGISTRY],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_proxy_config(
    path: Path | str,
    *,
    project_root: Path | str | None = None,
) -> ProxyFactorConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else config_path.parent.parent.resolve()
    )

    def resolve_path(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (
            (root / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )

    return ProxyFactorConfig(
        provider_uri=resolve_path(raw["provider_uri"]),
        output_root=resolve_path(raw["output_root"]),
        start_year=int(raw.get("start_year", 2008)),
        end_year=int(raw.get("end_year", 2025)),
        universe=str(raw.get("universe", "all")),
        frequency=str(raw.get("frequency", "day")),
        warmup_trading_days=int(raw.get("warmup_trading_days", 120)),
        factor_shard_size=int(raw.get("factor_shard_size", 32)),
        compression=str(raw.get("compression", "zstd")),
        min_global_valid_ratio=float(raw.get("min_global_valid_ratio", 0.05)),
        max_near_constant_ratio=float(raw.get("max_near_constant_ratio", 0.99)),
        project_root=root,
    )
