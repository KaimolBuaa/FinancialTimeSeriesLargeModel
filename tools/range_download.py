#!/usr/bin/env python3
"""Download a large immutable file in independently retried HTTP range chunks."""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def download(url: str, destination: Path, size: int, chunk_size: int, retries: int) -> None:
    partial = destination.with_name(destination.name + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as output:
        for start in range(0, size, chunk_size):
            end = min(start + chunk_size, size) - 1
            expected = end - start + 1
            for attempt in range(1, retries + 1):
                request = Request(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                        "User-Agent": "FactorPanel-FM-resource-bootstrap/1.0",
                    },
                )
                try:
                    with urlopen(request, timeout=90) as response:
                        payload = response.read()
                    if len(payload) != expected:
                        raise ValueError(f"chunk {start}-{end}: expected {expected} bytes, got {len(payload)}")
                    output.write(payload)
                    print(f"{end + 1}/{size} bytes", flush=True)
                    break
                except (HTTPError, URLError, OSError, ValueError) as error:
                    if attempt == retries:
                        raise RuntimeError(f"chunk {start}-{end} failed after {retries} attempts: {error}") from error
                    time.sleep(attempt * 2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    download(args.url, args.destination, args.size, args.chunk_size, args.retries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
