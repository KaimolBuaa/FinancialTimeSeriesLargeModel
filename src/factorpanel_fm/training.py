"""Stage A pretraining and Stage B supervised training modules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ._random import randperm_for_device as _randperm_for_device
from .batch import FactorPanelBatch
from .losses import (
    masked_huber_loss,
    negative_cross_sectional_ic_loss,
    pairwise_logistic_loss,
    quantile_pinball_loss,
)
from .model import EncoderOutput, FactorPanelEncoder
from .views import InputViews, build_input_views


def _validate_weight(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_horizons(name: str, values: Sequence[int]) -> tuple[int, ...]:
    horizons = tuple(values)
    if not horizons:
        raise ValueError(f"{name} must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in horizons):
        raise TypeError(f"{name} must contain integers")
    if any(value <= 0 for value in horizons):
        raise ValueError(f"{name} must contain positive values")
    if len(set(horizons)) != len(horizons):
        raise ValueError(f"{name} must not contain duplicates")
    return horizons


def _validate_quantiles(values: Sequence[float]) -> tuple[float, ...]:
    quantiles = tuple(values)
    if not quantiles:
        raise ValueError("quantiles must be nonempty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
        for value in quantiles
    ):
        raise ValueError("quantiles must contain finite real values in (0, 1)")
    if any(left >= right for left, right in zip(quantiles, quantiles[1:])):
        raise ValueError("quantiles must be strictly increasing")
    return quantiles


@dataclass(frozen=True)
class StageAConfig:
    """Objective and target configuration for Stage A pretraining."""

    mask_weight: float = 1.0
    future_weight: float = 0.5
    consistency_weight: float = 0.1
    future_horizons: tuple[int, ...] = (5, 20)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    mask_ratio: float = 0.4

    def __post_init__(self) -> None:
        for name in ("mask_weight", "future_weight", "consistency_weight"):
            _validate_weight(name, getattr(self, name))
        object.__setattr__(
            self,
            "future_horizons",
            _validate_horizons("future_horizons", self.future_horizons),
        )
        object.__setattr__(self, "quantiles", _validate_quantiles(self.quantiles))
        if isinstance(self.mask_ratio, bool) or not isinstance(self.mask_ratio, Real):
            raise TypeError("mask_ratio must be a real number")
        if not math.isfinite(float(self.mask_ratio)) or not 0.0 < self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be finite and in (0, 1]")


@dataclass(frozen=True)
class StageAOutput:
    total_loss: torch.Tensor
    mask_loss: torch.Tensor
    future_factor_loss: torch.Tensor
    consistency_loss: torch.Tensor
    encoder_output: EncoderOutput
    future_quantiles: torch.Tensor
    patch_reconstruction: torch.Tensor
    patch_mask: torch.Tensor
    encoder_patch_mask: torch.Tensor


def _patch_valid_from_views(views: InputViews, encoder: FactorPanelEncoder) -> torch.Tensor:
    return views.observed_mask.permute(0, 2, 1).unfold(
        2,
        encoder.config.patch_size,
        encoder.config.patch_stride,
    ).any(dim=-1)


def sample_patch_mask(
    patch_valid: torch.Tensor,
    mask_ratio: float = 0.4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample valid patches in shuffled contiguous spans at a target ratio."""

    if not isinstance(patch_valid, torch.Tensor):
        raise TypeError("patch_valid must be a torch.Tensor")
    if patch_valid.ndim != 3 or patch_valid.dtype != torch.bool:
        raise ValueError("patch_valid must be a bool tensor with shape [B, N, P]")
    if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, Real):
        raise TypeError("mask_ratio must be a real number")
    if not math.isfinite(float(mask_ratio)) or not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be finite and in (0, 1]")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    total_valid = int(patch_valid.sum().item())
    sampled = torch.zeros_like(patch_valid)
    if total_valid == 0:
        return sampled
    target_count = max(1, min(total_valid, int(math.floor(total_valid * mask_ratio + 0.5))))

    spans: list[tuple[int, int, tuple[int, ...]]] = []
    for batch_index in range(patch_valid.shape[0]):
        for asset_index in range(patch_valid.shape[1]):
            positions = patch_valid[batch_index, asset_index].nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                continue
            runs: list[list[int]] = []
            for position in positions.tolist():
                if not runs or position != runs[-1][-1] + 1:
                    runs.append([position])
                else:
                    runs[-1].append(position)
            for run in runs:
                for start in range(0, len(run), 2):
                    spans.append((batch_index, asset_index, tuple(run[start : start + 2])))

    order = _randperm_for_device(len(spans), patch_valid.device, generator=generator)
    remaining = target_count
    for span_index in order.tolist():
        batch_index, asset_index, positions = spans[span_index]
        chosen = positions[:remaining]
        sampled[batch_index, asset_index, list(chosen)] = True
        remaining -= len(chosen)
        if remaining == 0:
            break
    return sampled


def _patch_targets(views: InputViews, encoder: FactorPanelEncoder) -> tuple[torch.Tensor, torch.Tensor]:
    numeric = torch.stack((views.rank_gaussian, views.robust_z), dim=-1)
    targets = numeric.permute(0, 2, 1, 3).unfold(
        2,
        encoder.config.patch_size,
        encoder.config.patch_stride,
    ).permute(0, 1, 2, 4, 3)
    observed = views.observed_mask.permute(0, 2, 1).unfold(
        2,
        encoder.config.patch_size,
        encoder.config.patch_stride,
    )
    return targets, observed


def _expand_patch_mask_for_overlap(
    patch_mask: torch.Tensor,
    patch_size: int,
    patch_stride: int,
) -> torch.Tensor:
    """Mask every patch whose source interval overlaps a reconstruction target."""

    starts = torch.arange(patch_mask.shape[-1], device=patch_mask.device) * patch_stride
    ends = starts + patch_size
    overlaps = (starts[:, None] < ends[None, :]) & (starts[None, :] < ends[:, None])
    return (patch_mask.unsqueeze(-1) & overlaps).any(dim=-2)


def _connected_zero(*tensors: torch.Tensor) -> torch.Tensor:
    result = tensors[0].new_zeros(())
    for tensor in tensors:
        result = result + torch.where(torch.isfinite(tensor), tensor, 0.0).sum() * 0.0
    return result


class StageAModule(nn.Module):
    """Masked-view pretraining with future factors and view consistency."""

    def __init__(
        self,
        encoder: FactorPanelEncoder,
        config: StageAConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, FactorPanelEncoder):
            raise TypeError("encoder must be a FactorPanelEncoder")
        self.encoder = encoder
        self.config = config or StageAConfig()
        if not isinstance(self.config, StageAConfig):
            raise TypeError("config must be a StageAConfig")
        self.patch_reconstruction_head = nn.Linear(
            encoder.config.d_model,
            encoder.config.patch_size * 2,
        )
        self.future_head = nn.Linear(
            encoder.config.output_dim,
            len(self.config.future_horizons) * len(self.config.quantiles),
        )

    def _validate_targets(
        self,
        batch: FactorPanelBatch,
        future_targets: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> None:
        expected = (batch.batch_size, batch.num_assets, len(self.config.future_horizons))
        if not isinstance(future_targets, torch.Tensor) or future_targets.shape != expected:
            raise ValueError(f"future_targets must have shape {expected}")
        if not future_targets.is_floating_point():
            raise TypeError("future_targets must have a floating-point dtype")
        if not isinstance(future_mask, torch.Tensor) or future_mask.shape != expected:
            raise ValueError(f"future_mask must have shape {expected}")
        if future_mask.dtype != torch.bool:
            raise TypeError("future_mask must have bool dtype")
        if future_targets.device != batch.values.device or future_mask.device != batch.values.device:
            raise ValueError("future targets, mask, and batch must be on the same device")

    def forward(
        self,
        batch: FactorPanelBatch,
        future_targets: torch.Tensor,
        future_mask: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        second_batch: FactorPanelBatch | None = None,
        overlap_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> StageAOutput:
        self._validate_targets(batch, future_targets, future_mask)
        views = build_input_views(batch)
        patch_valid = _patch_valid_from_views(views, self.encoder)
        if patch_mask is None:
            selected_mask = sample_patch_mask(
                patch_valid,
                self.config.mask_ratio,
                generator=generator,
            )
        else:
            if not isinstance(patch_mask, torch.Tensor):
                raise TypeError("patch_mask must be a torch.Tensor")
            if patch_mask.shape != patch_valid.shape:
                raise ValueError(f"patch_mask must have shape {tuple(patch_valid.shape)}")
            if patch_mask.dtype != torch.bool:
                raise TypeError("patch_mask must have bool dtype")
            if patch_mask.device != batch.values.device:
                raise ValueError("patch_mask and batch must be on the same device")
            selected_mask = patch_mask & patch_valid

        encoder_patch_mask = _expand_patch_mask_for_overlap(
            selected_mask,
            self.encoder.config.patch_size,
            self.encoder.config.patch_stride,
        ) & patch_valid
        encoder_output = self.encoder(batch, views=views, patch_mask=encoder_patch_mask)
        reconstruction = self.patch_reconstruction_head(encoder_output.patch_states).unflatten(
            -1,
            (self.encoder.config.patch_size, 2),
        )
        patch_targets, patch_observed = _patch_targets(views, self.encoder)
        reconstruction_mask = selected_mask.unsqueeze(-1) & patch_observed
        mask_loss = masked_huber_loss(
            reconstruction,
            patch_targets,
            reconstruction_mask.unsqueeze(-1).expand_as(reconstruction),
        )

        future_quantiles = self.future_head(encoder_output.features).unflatten(
            -1,
            (len(self.config.future_horizons), len(self.config.quantiles)),
        )
        future_factor_loss = quantile_pinball_loss(
            future_quantiles,
            future_targets,
            future_mask,
            quantiles=self.config.quantiles,
        )

        if second_batch is None:
            if overlap_mask is not None:
                raise ValueError("overlap_mask requires second_batch")
            consistency_loss = _connected_zero(encoder_output.features)
        else:
            if not isinstance(second_batch, FactorPanelBatch):
                raise TypeError("second_batch must be a FactorPanelBatch")
            if second_batch.batch_size != batch.batch_size:
                raise ValueError("second_batch must match batch size")
            if second_batch.dates.shape != batch.dates.shape or not torch.equal(
                second_batch.dates,
                batch.dates,
            ):
                raise ValueError("second_batch dates must exactly match batch dates")
            for name, asset_ids in (
                ("batch", batch.asset_ids),
                ("second_batch", second_batch.asset_ids),
            ):
                sorted_ids = asset_ids.sort(dim=-1).values
                if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item():
                    raise ValueError(f"{name} asset_ids must be unique within each batch")
            second_output = self.encoder(second_batch)
            matches = batch.asset_ids.unsqueeze(-1) == second_batch.asset_ids.unsqueeze(-2)
            feature_device = second_output.features.device
            has_match = matches.any(dim=-1).to(feature_device)
            second_indices = matches.to(torch.int64).argmax(dim=-1)
            feature_indices = second_indices.to(feature_device)
            matched_features = second_output.features.gather(
                1,
                feature_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    second_output.features.shape[-1],
                ),
            )
            matched_valid = second_output.asset_valid.gather(1, feature_indices)
            overlap = encoder_output.asset_valid & has_match & matched_valid
            if overlap_mask is not None:
                if not isinstance(overlap_mask, torch.Tensor):
                    raise TypeError("overlap_mask must be a torch.Tensor")
                if overlap_mask.shape != overlap.shape:
                    raise ValueError(f"overlap_mask must have shape {tuple(overlap.shape)}")
                if overlap_mask.dtype != torch.bool:
                    raise TypeError("overlap_mask must have bool dtype")
                if overlap_mask.device != batch.values.device:
                    raise ValueError("overlap_mask and batch must be on the same device")
                overlap = overlap & overlap_mask
            if overlap.any().item():
                similarity = F.cosine_similarity(
                    encoder_output.features[overlap],
                    matched_features[overlap],
                    dim=-1,
                )
                consistency_loss = (1.0 - similarity).mean()
            else:
                consistency_loss = _connected_zero(
                    encoder_output.features,
                    second_output.features,
                )

        total_loss = (
            self.config.mask_weight * mask_loss
            + self.config.future_weight * future_factor_loss
            + self.config.consistency_weight * consistency_loss
            + _connected_zero(encoder_output.factor_embedding, self.encoder.mask_token)
        )
        return StageAOutput(
            total_loss=total_loss,
            mask_loss=mask_loss,
            future_factor_loss=future_factor_loss,
            consistency_loss=consistency_loss,
            encoder_output=encoder_output,
            future_quantiles=future_quantiles,
            patch_reconstruction=reconstruction,
            patch_mask=selected_mask,
            encoder_patch_mask=encoder_patch_mask,
        )


@dataclass(frozen=True)
class StageBConfig:
    """Objective and optimization configuration for Stage B fine-tuning."""

    ic_weight: float = 1.0
    pairwise_weight: float = 0.5
    huber_weight: float = 0.2
    horizons: tuple[int, ...] = (1, 5, 20)
    initial_freeze_steps: int = 5000
    base_lr: float = 3e-4
    unfreeze_lr_scale: float = 0.1

    def __post_init__(self) -> None:
        for name in ("ic_weight", "pairwise_weight", "huber_weight"):
            _validate_weight(name, getattr(self, name))
        object.__setattr__(self, "horizons", _validate_horizons("horizons", self.horizons))
        if isinstance(self.initial_freeze_steps, bool) or not isinstance(
            self.initial_freeze_steps, Integral
        ):
            raise TypeError("initial_freeze_steps must be an integer")
        if self.initial_freeze_steps < 0:
            raise ValueError("initial_freeze_steps must be nonnegative")
        for name in ("base_lr", "unfreeze_lr_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class StageBOutput:
    total_loss: torch.Tensor
    ic_loss: torch.Tensor
    pairwise_loss: torch.Tensor
    huber_loss: torch.Tensor
    encoder_output: EncoderOutput
    return_scores: torch.Tensor

    @property
    def scores(self) -> torch.Tensor:
        return self.return_scores

    @property
    def return_predictions(self) -> torch.Tensor:
        return self.return_scores


class StageBModule(nn.Module):
    """Supervised cross-sectional return prediction module."""

    def __init__(
        self,
        encoder: FactorPanelEncoder,
        config: StageBConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, FactorPanelEncoder):
            raise TypeError("encoder must be a FactorPanelEncoder")
        self.encoder = encoder
        self.config = config or StageBConfig()
        if not isinstance(self.config, StageBConfig):
            raise TypeError("config must be a StageBConfig")
        self.return_head = nn.Linear(encoder.config.output_dim, len(self.config.horizons))

    def forward(
        self,
        batch: FactorPanelBatch,
        targets: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> StageBOutput:
        expected = (batch.batch_size, batch.num_assets, len(self.config.horizons))
        if not isinstance(targets, torch.Tensor) or targets.shape != expected:
            raise ValueError(f"targets must have shape {expected}")
        if not targets.is_floating_point():
            raise TypeError("targets must have a floating-point dtype")
        if not isinstance(mask, torch.Tensor) or mask.shape != expected:
            raise ValueError(f"mask must have shape {expected}")
        if mask.dtype != torch.bool:
            raise TypeError("mask must have bool dtype")
        if targets.device != batch.values.device or mask.device != batch.values.device:
            raise ValueError("targets, mask, and batch must be on the same device")

        encoder_output = self.encoder(batch)
        return_scores = self.return_head(encoder_output.features)
        ic_loss = negative_cross_sectional_ic_loss(return_scores, targets, mask)
        pairwise_loss = pairwise_logistic_loss(
            return_scores,
            targets,
            mask,
            generator=generator,
        )
        huber_loss = masked_huber_loss(return_scores, targets, mask)
        total_loss = (
            self.config.ic_weight * ic_loss
            + self.config.pairwise_weight * pairwise_loss
            + self.config.huber_weight * huber_loss
            + _connected_zero(encoder_output.factor_embedding, self.encoder.mask_token)
        )
        return StageBOutput(
            total_loss=total_loss,
            ic_loss=ic_loss,
            pairwise_loss=pairwise_loss,
            huber_loss=huber_loss,
            encoder_output=encoder_output,
            return_scores=return_scores,
        )


def _lower_temporal_prefixes(module: StageBModule) -> tuple[str, ...]:
    lower_count = len(module.encoder.temporal_blocks) // 2
    return tuple(f"encoder.temporal_blocks.{index}." for index in range(lower_count))


def configure_stage_b_trainability(module: StageBModule, step: int) -> None:
    """Keep every Stage B parameter in the graph across the freeze boundary."""

    if not isinstance(module, StageBModule):
        raise TypeError("module must be a StageBModule")
    if isinstance(step, bool) or not isinstance(step, Integral):
        raise TypeError("step must be an integer")
    if step < 0:
        raise ValueError("step must be nonnegative")
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def build_stage_b_optimizer(
    module: StageBModule,
    step: int,
    weight_decay: float = 0.05,
) -> torch.optim.AdamW:
    """Build deduplicated decay and learning-rate groups for Stage B."""

    if isinstance(weight_decay, bool) or not isinstance(weight_decay, Real):
        raise TypeError("weight_decay must be a real number")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    configure_stage_b_trainability(module, step)
    frozen = step < module.config.initial_freeze_steps
    lower_prefixes = _lower_temporal_prefixes(module)
    grouped: dict[tuple[bool, float], list[nn.Parameter]] = {}
    seen: set[int] = set()
    for name, parameter in module.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise RuntimeError(f"duplicate optimizer parameter: {name}")
        seen.add(parameter_id)
        is_lower = name.startswith(lower_prefixes)
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        decay = 0.0 if no_decay else float(weight_decay)
        grouped.setdefault((is_lower, decay), []).append(parameter)
    parameter_groups = [
        {
            "params": parameters,
            "lr": (
                0.0
                if is_lower and frozen
                else module.config.base_lr
                * (module.config.unfreeze_lr_scale if is_lower else 1.0)
            ),
            "weight_decay": decay,
            "stage_b_lower": is_lower,
        }
        for (is_lower, decay), parameters in grouped.items()
    ]
    if not parameter_groups:
        raise ValueError("module has no trainable parameters")
    return torch.optim.AdamW(parameter_groups, lr=module.config.base_lr)


def update_stage_b_optimizer(
    module: StageBModule,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> torch.optim.Optimizer:
    """Update Stage B freeze state and group learning rates without rebuilding."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    configure_stage_b_trainability(module, step)
    expected_ids = {id(parameter) for parameter in module.parameters()}
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != expected_ids:
        raise ValueError("optimizer parameters must exactly match the Stage B module")
    frozen = step < module.config.initial_freeze_steps
    for group in optimizer.param_groups:
        is_lower = group.get("stage_b_lower")
        if not isinstance(is_lower, bool):
            raise ValueError("optimizer groups must be created by build_stage_b_optimizer")
        group["lr"] = (
            0.0
            if is_lower and frozen
            else module.config.base_lr
            * (module.config.unfreeze_lr_scale if is_lower else 1.0)
        )
    return optimizer


__all__ = [
    "StageAConfig",
    "StageAModule",
    "StageAOutput",
    "StageBConfig",
    "StageBModule",
    "StageBOutput",
    "build_stage_b_optimizer",
    "configure_stage_b_trainability",
    "sample_patch_mask",
    "update_stage_b_optimizer",
]
