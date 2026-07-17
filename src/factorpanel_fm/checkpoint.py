"""Atomic model and optimizer checkpoint persistence."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointInfo:
    """Training position and metadata restored from a checkpoint."""

    step: int
    metadata: dict[str, Any]


def save_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save model, optimizer, training step, and metadata."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if isinstance(step, bool) or not isinstance(step, Integral):
        raise TypeError("step must be an integer")
    if step < 0:
        raise ValueError("step must be nonnegative")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": int(step),
        "metadata": dict(metadata or {}),
    }
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointInfo:
    """Restore a checkpoint and return its training position metadata."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer or None")
    if not isinstance(strict, bool):
        raise TypeError("strict must be bool")
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    required = {"model_state", "optimizer_state", "step", "metadata"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    step = payload["step"]
    metadata = payload["metadata"]
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return CheckpointInfo(step=int(step), metadata=dict(metadata))


__all__ = ["CheckpointInfo", "load_checkpoint", "save_checkpoint"]
