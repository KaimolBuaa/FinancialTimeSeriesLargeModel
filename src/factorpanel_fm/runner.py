"""Executable local training runtime for FactorPanel-FM stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import os
from pathlib import Path
import random
import time
from typing import Callable, Iterable, Iterator

import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .data import FactorPanelSample
from .training import (
    StageAModule,
    StageBModule,
    build_stage_b_optimizer,
    update_stage_b_optimizer,
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Optimization and execution settings for a bounded local run."""

    stage: str
    model: str = "tiny"
    max_steps: int = 2
    micro_batch_size: int = 1
    gradient_accumulation: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42
    bf16: bool = True
    checkpoint_every: int = 1000
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.stage not in ("a", "b"):
            raise ValueError("stage must be 'a' or 'b'")
        if self.model not in ("tiny", "small"):
            raise ValueError("model must be 'tiny' or 'small'")
        for name in (
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation",
            "checkpoint_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        for name in ("lr", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.weight_decay, bool) or not isinstance(
            self.weight_decay, Real
        ):
            raise TypeError("weight_decay must be a real number")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if isinstance(self.warmup_ratio, bool) or not isinstance(
            self.warmup_ratio, Real
        ):
            raise TypeError("warmup_ratio must be a real number")
        if (
            not math.isfinite(float(self.warmup_ratio))
            or not 0 <= self.warmup_ratio <= 1
        ):
            raise ValueError("warmup_ratio must be finite and in [0, 1]")
        if not isinstance(self.bf16, bool):
            raise TypeError("bf16 must be bool")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be auto, cpu, mps, or cuda")


@dataclass(frozen=True)
class RunSummary:
    """Machine-readable result of a local runtime invocation.

    ``micro_steps`` and ``consumed_samples`` are cumulative across resumed runs.
    """

    stage: str
    model: str
    start_step: int
    step: int
    micro_steps: int
    consumed_samples: int
    final_loss: float
    device: str
    param_count: int
    trainable_param_count: int
    bf16_enabled: bool
    peak_memory_bytes: int | None
    elapsed_seconds: float
    checkpoint_path: str | None


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit or preferred available Torch device."""

    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return torch.device(requested)


def _bf16_supported(device: torch.device, requested: bool) -> bool:
    if not requested or device.type != "cuda":
        return False
    return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())


def warmup_cosine_multiplier(
    step: int, *, warmup_steps: int, total_steps: int
) -> float:
    """Return a linear-warmup then cosine-decay learning-rate multiplier."""

    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (step, warmup_steps, total_steps)
    ):
        raise TypeError("scheduler positions must be integers")
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps > total_steps or step < 0:
        raise ValueError("invalid scheduler positions")
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return 0.0 if step >= total_steps - 1 else 1.0
    progress = min(max((step - warmup_steps) / (decay_steps - 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def auto_micro_batch_probe(
    requested: int,
    probe: Callable[[int], None],
    *,
    device: torch.device,
) -> int:
    """Find the largest working size up to ``requested`` using an injected CUDA probe."""

    if isinstance(requested, bool) or not isinstance(requested, Integral):
        raise TypeError("requested must be an integer")
    if requested <= 0:
        raise ValueError("requested must be positive")
    if not callable(probe):
        raise TypeError("probe must be callable")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError("automatic micro-batch probing is CUDA-only")
    for size in range(int(requested), 0, -1):
        try:
            probe(size)
            return size
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("CUDA probe failed even at micro-batch size 1")


class _CyclingSamples:
    def __init__(
        self, source: Iterable[FactorPanelSample] | Iterator[FactorPanelSample]
    ) -> None:
        self.source = source
        self.iterator = iter(source)
        self.is_iterator = self.iterator is source
        self.cache: list[FactorPanelSample] = []
        self.cache_index = 0
        self.exhausted = False

    def next(self) -> FactorPanelSample:
        if not self.exhausted:
            try:
                sample = next(self.iterator)
                if self.is_iterator:
                    self.cache.append(sample)
                return self._validate(sample)
            except StopIteration:
                self.exhausted = self.is_iterator
        if self.is_iterator:
            if not self.cache:
                raise ValueError("training data is empty")
            sample = self.cache[self.cache_index]
            self.cache_index = (self.cache_index + 1) % len(self.cache)
            return self._validate(sample)
        self.iterator = iter(self.source)
        try:
            return self._validate(next(self.iterator))
        except StopIteration as error:
            raise ValueError("training data is empty") from error

    @staticmethod
    def _validate(sample: object) -> FactorPanelSample:
        if not isinstance(sample, FactorPanelSample):
            raise TypeError("training data must yield FactorPanelSample values")
        return sample


def _validate_stage_module(module: nn.Module, stage: str) -> None:
    expected = StageAModule if stage == "a" else StageBModule
    if not isinstance(module, expected):
        raise TypeError(f"stage {stage} requires {expected.__name__}")


def _rng_metadata() -> dict[str, object]:
    cuda_states = (
        [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    mps_state = (
        torch.mps.get_rng_state().cpu().tolist()
        if torch.backends.mps.is_available()
        else None
    )
    return {
        "python_rng_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu().tolist(),
        "cuda_rng_states": cuda_states,
        "mps_rng_state": mps_state,
    }


def _as_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_as_tuple(item) for item in value)
    return value


def _restore_rng(metadata: dict[str, object]) -> None:
    python_state = metadata.get("python_rng_state")
    cpu_state = metadata.get("torch_cpu_rng_state")
    cuda_states = metadata.get("cuda_rng_states")
    mps_state = metadata.get("mps_rng_state")
    if not isinstance(python_state, (list, tuple)):
        raise ValueError("checkpoint is missing python_rng_state")
    if not isinstance(cpu_state, list) or not all(
        isinstance(value, int) for value in cpu_state
    ):
        raise ValueError("checkpoint is missing torch_cpu_rng_state")
    if not isinstance(cuda_states, list):
        raise ValueError("checkpoint cuda_rng_states must be a list")
    if mps_state is not None and not isinstance(mps_state, list):
        raise ValueError("checkpoint mps_rng_state must be a list or null")
    random.setstate(_as_tuple(python_state))
    torch.set_rng_state(torch.tensor(cpu_state, dtype=torch.uint8))
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(
            [torch.tensor(state, dtype=torch.uint8) for state in cuda_states]
        )
    if torch.backends.mps.is_available() and mps_state is not None:
        torch.mps.set_rng_state(torch.tensor(mps_state, dtype=torch.uint8))


def _checkpoint_metadata(
    config: RuntimeConfig,
    *,
    consumed_samples: int,
    micro_steps_total: int,
) -> dict[str, object]:
    return {
        "stage": config.stage,
        "runtime_config": asdict(config),
        "consumed_samples": consumed_samples,
        "micro_steps_total": micro_steps_total,
        **_rng_metadata(),
    }


def _checkpoint_stage(path: Path) -> str | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    stage = metadata.get("stage")
    return stage if isinstance(stage, str) else None


def _set_optimizer_step_lrs(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: RuntimeConfig,
    *,
    step: int,
) -> None:
    """Compose the stage base/freeze rates with the runtime schedule."""

    warmup_steps = min(
        config.max_steps, math.ceil(config.max_steps * config.warmup_ratio)
    )
    multiplier = warmup_cosine_multiplier(
        step,
        warmup_steps=warmup_steps,
        total_steps=config.max_steps,
    )
    if config.stage == "b":
        if not isinstance(module, StageBModule):
            raise TypeError("stage b requires StageBModule")
        update_stage_b_optimizer(module, optimizer, step)
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * multiplier
        return
    for group in optimizer.param_groups:
        group["lr"] = float(config.lr) * multiplier


def run_training(
    module: nn.Module,
    data: Iterable[FactorPanelSample] | Iterator[FactorPanelSample],
    config: RuntimeConfig,
    checkpoint_path: str | os.PathLike[str] | None = None,
    resume: bool = False,
) -> RunSummary:
    """Run a deterministic, bounded Stage A or Stage B optimization loop."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch module")
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be a RuntimeConfig")
    if not isinstance(resume, bool):
        raise TypeError("resume must be bool")
    _validate_stage_module(module, config.stage)
    destination = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and destination is None:
        raise ValueError("resume requires checkpoint_path")
    if resume and not destination.is_file():
        raise ValueError(f"checkpoint_path does not exist: {destination}")
    samples = _CyclingSamples(data)
    if resume and samples.is_iterator:
        raise ValueError(
            "resume does not support a one-shot iterator; provide a replayable iterable"
        )

    device = resolve_device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    module.to(device)
    if isinstance(module, StageBModule):
        if module.config.base_lr != float(config.lr):
            raise ValueError(
                "RuntimeConfig.lr must equal StageBConfig.base_lr so freeze groups stay coherent"
            )
        optimizer = build_stage_b_optimizer(
            module,
            step=0,
            weight_decay=float(config.weight_decay),
        )
    else:
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )

    start_step = 0
    consumed_samples = 0
    micro_steps_total = 0
    resume_metadata: dict[str, object] | None = None
    if resume:
        assert destination is not None
        checkpoint_stage = _checkpoint_stage(destination)
        if checkpoint_stage != config.stage:
            raise ValueError(
                f"checkpoint stage {checkpoint_stage!r} does not match runtime stage {config.stage!r}"
            )
        info = load_checkpoint(
            destination, module, optimizer=optimizer, map_location=device
        )
        start_step = info.step
        if start_step >= config.max_steps:
            raise ValueError("checkpoint step must be less than max_steps")
        resume_metadata = info.metadata
        consumed_value = resume_metadata.get("consumed_samples")
        micro_steps_value = resume_metadata.get("micro_steps_total")
        if isinstance(consumed_value, bool) or not isinstance(consumed_value, Integral):
            raise ValueError("checkpoint consumed_samples must be an integer")
        if isinstance(micro_steps_value, bool) or not isinstance(
            micro_steps_value, Integral
        ):
            raise ValueError("checkpoint micro_steps_total must be an integer")
        if consumed_value < 0 or micro_steps_value < 0:
            raise ValueError("checkpoint sample counters must be nonnegative")
        consumed_samples = int(consumed_value)
        micro_steps_total = int(micro_steps_value)
        for _ in range(consumed_samples):
            samples.next()
        _restore_rng(resume_metadata)

    bf16_enabled = _bf16_supported(device, config.bf16)
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    final_loss = float("nan")
    started = time.perf_counter()
    while step < config.max_steps:
        _set_optimizer_step_lrs(module, optimizer, config, step=step)
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation):
            sample = samples.next()
            consumed_samples += 1
            if sample.batch.batch_size > config.micro_batch_size:
                raise ValueError(
                    "sample batch size exceeds configured micro_batch_size"
                )
            sample = sample.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16_enabled,
            ):
                if config.stage == "a":
                    assert isinstance(module, StageAModule)
                    output = module(
                        sample.batch,
                        sample.future_factor_targets,
                        sample.future_factor_mask,
                    )
                else:
                    assert isinstance(module, StageBModule)
                    output = module(
                        sample.batch,
                        sample.return_targets,
                        sample.return_mask,
                    )
                loss = output.total_loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("training loss is not finite")
            (loss / config.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach().cpu())
            micro_steps_total += 1
        torch.nn.utils.clip_grad_norm_(module.parameters(), float(config.grad_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        final_loss = accumulated_loss / config.gradient_accumulation
        if destination is not None and step % config.checkpoint_every == 0:
            save_checkpoint(
                destination,
                module,
                optimizer,
                step,
                metadata=_checkpoint_metadata(
                    config,
                    consumed_samples=consumed_samples,
                    micro_steps_total=micro_steps_total,
                ),
            )

    if destination is not None and (
        step % config.checkpoint_every != 0 or not destination.exists()
    ):
        save_checkpoint(
            destination,
            module,
            optimizer,
            step,
            metadata=_checkpoint_metadata(
                config,
                consumed_samples=consumed_samples,
                micro_steps_total=micro_steps_total,
            ),
        )
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return RunSummary(
        stage=config.stage,
        model=config.model,
        start_step=start_step,
        step=step,
        micro_steps=micro_steps_total,
        consumed_samples=consumed_samples,
        final_loss=final_loss,
        device=str(device),
        param_count=sum(parameter.numel() for parameter in module.parameters()),
        trainable_param_count=sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
        bf16_enabled=bf16_enabled,
        peak_memory_bytes=peak_memory,
        elapsed_seconds=elapsed,
        checkpoint_path=str(destination) if destination is not None else None,
    )


__all__ = [
    "RunSummary",
    "RuntimeConfig",
    "auto_micro_batch_probe",
    "resolve_device",
    "run_training",
    "warmup_cosine_multiplier",
]
