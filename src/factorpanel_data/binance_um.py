"""Verified, storage-bounded downloads for Binance USD-M monthly K-lines."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
import os


BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"


class StorageLimitError(RuntimeError):
    """Raised before a download would exceed the configured storage budget."""


def _validate_month(month: str) -> None:
    datetime.strptime(month, "%Y-%m")


def archive_url(symbol: str, month: str) -> str:
    """Return the immutable official monthly 1m archive location."""

    _validate_month(month)
    return f"{BASE_URL}/{symbol}/1m/{symbol}-1m-{month}.zip"


def archive_path(data_root: Path, symbol: str, month: str) -> Path:
    """Return the local raw archive location for one symbol-month."""

    _validate_month(month)
    year = month[:4]
    return data_root / "raw_1m" / symbol / year / f"{symbol}-1m-{month}.zip"


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def ensure_storage_budget(
    *,
    current_bytes: int,
    incoming_bytes: int,
    target_bytes: int,
    hard_bytes: int,
    reserve_ratio: float = 0.20,
) -> None:
    """Reject a download before writes when its retained and working size is unsafe."""

    if min(current_bytes, incoming_bytes, target_bytes, hard_bytes) < 0:
        raise ValueError("storage values must be non-negative")
    if target_bytes > hard_bytes:
        raise ValueError("target_bytes must not exceed hard_bytes")
    if not 0 <= reserve_ratio < 1:
        raise ValueError("reserve_ratio must be in [0, 1)")
    projected = current_bytes + incoming_bytes
    working_set = projected + int(incoming_bytes * reserve_ratio)
    if projected > hard_bytes:
        raise StorageLimitError(
            f"hard storage limit exceeded: projected={projected} hard={hard_bytes}"
        )
    if working_set > target_bytes:
        raise StorageLimitError(
            f"target storage budget exceeded: working_set={working_set} target={target_bytes}"
        )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_from_text(text: str) -> str:
    fields = text.strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError("invalid Binance checksum file")
    return fields[0].lower()


def _request_bytes(url: str) -> tuple[bytes, int | None]:
    request = Request(url, headers={"User-Agent": "FactorPanel-FM/0.1"})
    with urlopen(request, timeout=90) as response:
        length = response.headers.get("Content-Length")
        return response.read(), int(length) if length else None


def download_verified_archive(
    *,
    data_root: Path,
    symbol: str,
    month: str,
    target_bytes: int = 60 * 1024**3,
    hard_bytes: int = 80 * 1024**3,
) -> Path:
    """Download one archive only when it fits budget and matches Binance checksum."""

    destination = archive_path(data_root, symbol, month)
    checksum_destination = destination.with_suffix(".zip.CHECKSUM")
    checksum_text, _ = _request_bytes(f"{archive_url(symbol, month)}.CHECKSUM")
    expected_checksum = _checksum_from_text(checksum_text.decode("utf-8"))
    if destination.exists() and _sha256(destination) == expected_checksum:
        if not checksum_destination.exists():
            checksum_destination.write_bytes(checksum_text)
        return destination

    request = Request(archive_url(symbol, month), headers={"User-Agent": "FactorPanel-FM/0.1"})
    with urlopen(request, timeout=90) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is None:
            raise RuntimeError("archive response did not provide Content-Length")
        incoming_bytes = int(content_length)
        ensure_storage_budget(
            current_bytes=_directory_size(data_root),
            incoming_bytes=incoming_bytes,
            target_bytes=target_bytes,
            hard_bytes=hard_bytes,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".zip.part")
        digest = sha256()
        with partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if digest.hexdigest() != expected_checksum:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {symbol} {month}")
    partial.replace(destination)
    checksum_destination.write_bytes(checksum_text)
    return destination
