"""Atomic model and optimizer checkpoint persistence."""

from __future__ import annotations

from dataclasses import dataclass
import copy
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


def _validate_metadata(value: object, path: str = "metadata") -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_metadata(item, f"{path}.{key}")
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _validate_metadata(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _preflight_model_state(
    model: nn.Module,
    model_state: object,
    strict: bool,
) -> None:
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model_state must be a mapping")
    current_state = model.state_dict()
    checkpoint_keys = set(model_state)
    current_keys = set(current_state)
    if not all(isinstance(key, str) for key in checkpoint_keys):
        raise ValueError("checkpoint model_state keys must be strings")
    if strict and checkpoint_keys != current_keys:
        missing = sorted(current_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - current_keys)
        raise ValueError(
            f"checkpoint model keys do not match; missing={missing}, unexpected={unexpected}"
        )
    for key in current_keys & checkpoint_keys:
        checkpoint_value = model_state[key]
        current_value = current_state[key]
        if not isinstance(checkpoint_value, torch.Tensor):
            raise ValueError(f"checkpoint model value {key!r} must be a tensor")
        if checkpoint_value.shape != current_value.shape:
            raise ValueError(
                f"checkpoint model shape mismatch for {key!r}: "
                f"{tuple(checkpoint_value.shape)} != {tuple(current_value.shape)}"
            )
        if checkpoint_value.dtype != current_value.dtype:
            raise ValueError(
                f"checkpoint model dtype mismatch for {key!r}: "
                f"{checkpoint_value.dtype} != {current_value.dtype}"
            )


def _preflight_optimizer_state(
    optimizer: torch.optim.Optimizer,
    optimizer_state: object,
) -> None:
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer_state must be a mapping")
    checkpoint_groups = optimizer_state.get("param_groups")
    current_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(checkpoint_groups, list) or not isinstance(current_groups, list):
        raise ValueError("checkpoint optimizer param groups must be lists")
    if len(checkpoint_groups) != len(current_groups):
        raise ValueError("checkpoint optimizer param groups have incompatible counts")
    for index, (checkpoint_group, current_group) in enumerate(
        zip(checkpoint_groups, current_groups)
    ):
        if not isinstance(checkpoint_group, Mapping):
            raise ValueError(f"checkpoint optimizer param group {index} must be a mapping")
        checkpoint_params = checkpoint_group.get("params")
        current_params = current_group.get("params")
        if not isinstance(checkpoint_params, list) or not isinstance(current_params, list):
            raise ValueError("checkpoint optimizer param groups must contain parameter lists")
        if len(checkpoint_params) != len(current_params):
            raise ValueError(
                f"checkpoint optimizer param groups have incompatible params at group {index}"
            )


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
    _validate_metadata(metadata or {})

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
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
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
    _validate_metadata(metadata)
    _preflight_model_state(model, payload["model_state"], strict)
    if optimizer is not None:
        _preflight_optimizer_state(optimizer, payload["optimizer_state"])

    original_model_state = copy.deepcopy(model.state_dict())
    original_optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    try:
        model.load_state_dict(payload["model_state"], strict=strict)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
    except Exception:
        model.load_state_dict(original_model_state, strict=True)
        if optimizer is not None and original_optimizer_state is not None:
            optimizer.load_state_dict(original_optimizer_state)
        raise
    return CheckpointInfo(step=int(step), metadata=dict(metadata))


__all__ = ["CheckpointInfo", "load_checkpoint", "save_checkpoint"]
