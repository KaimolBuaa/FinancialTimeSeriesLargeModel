"""Device-compatible random tensor helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _generator_device(
    target_device: torch.device,
    generator: torch.Generator | None,
) -> torch.device:
    return target_device if generator is None else torch.device(generator.device)


def randperm_for_device(
    size: int,
    target_device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    random_device = _generator_device(target_device, generator)
    return torch.randperm(size, device=random_device, generator=generator).to(target_device)


def randint_for_device(
    high: int,
    size: Sequence[int],
    target_device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    random_device = _generator_device(target_device, generator)
    return torch.randint(
        high,
        tuple(size),
        device=random_device,
        generator=generator,
    ).to(target_device)
