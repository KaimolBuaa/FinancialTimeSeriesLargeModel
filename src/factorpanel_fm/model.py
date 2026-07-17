"""FactorPanel-FM Small encoder architecture."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from .batch import FactorPanelBatch
from .views import InputViews, build_input_views


@dataclass(frozen=True)
class ModelConfig:
    """Architecture configuration for a FactorPanel encoder."""

    context_length: int = 256
    input_channels: int = 3
    patch_size: int = 16
    patch_stride: int = 8
    d_model: int = 384
    temporal_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 1536
    num_latents: int = 32
    set_layers: int = 2
    output_dim: int = 128
    dropout: float = 0.1
    use_set_mixer: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            "context_length",
            "input_channels",
            "patch_size",
            "patch_stride",
            "d_model",
            "temporal_layers",
            "num_heads",
            "ffn_dim",
            "num_latents",
            "set_layers",
            "output_dim",
        )
        for name in dimensions:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.context_length < self.patch_size:
            raise ValueError("context_length must be at least patch_size")
        if (self.context_length - self.patch_size) % self.patch_stride != 0:
            raise ValueError("patches must exactly cover context_length")
        if self.input_channels != 3:
            raise ValueError("input_channels must be 3 for the current input view contract")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, Real):
            raise TypeError("dropout must be a real number")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(self.use_set_mixer, bool):
            raise TypeError("use_set_mixer must be bool")

    @property
    def num_patches(self) -> int:
        return (self.context_length - self.patch_size) // self.patch_stride + 1

    @classmethod
    def small(cls, **overrides: object) -> ModelConfig:
        """Construct the production Small configuration."""

        return cls(**overrides)

    @classmethod
    def tiny(cls, **overrides: object) -> ModelConfig:
        """Construct a compact configuration suitable for tests and smoke runs."""

        values: dict[str, object] = {
            "context_length": 16,
            "input_channels": 3,
            "patch_size": 4,
            "patch_stride": 4,
            "d_model": 32,
            "temporal_layers": 2,
            "num_heads": 4,
            "ffn_dim": 64,
            "num_latents": 4,
            "set_layers": 1,
            "output_dim": 16,
            "dropout": 0.0,
            "use_set_mixer": True,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class EncoderOutput:
    """Public encoder result and its reusable intermediate states."""

    features: torch.Tensor
    factor_embedding: torch.Tensor
    temporal_states: torch.Tensor
    patch_states: torch.Tensor
    patch_valid: torch.Tensor
    asset_valid: torch.Tensor

    def __post_init__(self) -> None:
        for field in fields(self):
            if not isinstance(getattr(self, field.name), torch.Tensor):
                raise TypeError(f"{field.name} must be a torch.Tensor")
        if self.features.ndim != 3:
            raise ValueError("features must have shape [B, N, output_dim]")
        batch_size, num_assets, output_dim = self.features.shape
        if self.factor_embedding.shape != (batch_size, output_dim):
            raise ValueError("factor_embedding must have shape [B, output_dim]")
        if self.temporal_states.ndim != 3:
            raise ValueError("temporal_states must have shape [B, N, d_model]")
        if self.temporal_states.shape[:2] != (batch_size, num_assets):
            raise ValueError("temporal_states must match feature batch and assets")
        if self.patch_states.ndim != 4:
            raise ValueError("patch_states must have shape [B, N, P, d_model]")
        if self.patch_states.shape[:2] != (batch_size, num_assets):
            raise ValueError("patch_states must match feature batch and assets")
        if self.patch_states.shape[-1] != self.temporal_states.shape[-1]:
            raise ValueError("patch and temporal states must share d_model")
        patch_shape = self.patch_states.shape[:3]
        if self.patch_valid.shape != patch_shape:
            raise ValueError("patch_valid must have shape [B, N, P]")
        if self.asset_valid.shape != (batch_size, num_assets):
            raise ValueError("asset_valid must have shape [B, N]")
        if self.patch_valid.dtype != torch.bool or self.asset_valid.dtype != torch.bool:
            raise TypeError("validity masks must have bool dtype")


def _apply_rope(tensor: torch.Tensor) -> torch.Tensor:
    sequence_length = tensor.shape[-2]
    head_dim = tensor.shape[-1]
    positions = torch.arange(sequence_length, device=tensor.device, dtype=torch.float32)
    frequencies = torch.arange(0, head_dim, 2, device=tensor.device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10_000.0) * frequencies / head_dim)
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos().to(tensor.dtype)[None, None]
    sine = angles.sin().to(tensor.dtype)[None, None]
    even, odd = tensor[..., 0::2], tensor[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)


class _MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, *, use_rope: bool = False) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout
        self.use_rope = use_rope
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_valid: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))
        if self.use_rope:
            q = _apply_rope(q)
            k = _apply_rope(k)

        any_key = key_valid.any(dim=-1)
        safe_key_valid = key_valid.clone()
        safe_key_valid[:, 0] |= ~any_key
        attention_mask = safe_key_valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).flatten(-2)
        return self.out_proj(attended) * query_valid.unsqueeze(-1)


class _FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class _TemporalBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = _MultiHeadAttention(config, use_rope=True)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = _FeedForward(config)

    def forward(self, states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        states = states + self.attention_dropout(
            self.attention(
                self.attention_norm(states),
                self.attention_norm(states),
                valid,
                valid,
            )
        )
        states = states * valid.unsqueeze(-1)
        states = states + self.ffn(self.ffn_norm(states))
        return states * valid.unsqueeze(-1)


class _InducedSetBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.empty(config.num_latents, config.d_model))
        nn.init.normal_(self.latents, std=config.d_model**-0.5)

        self.latent_query_norm = nn.LayerNorm(config.d_model)
        self.asset_key_norm = nn.LayerNorm(config.d_model)
        self.latent_attention = _MultiHeadAttention(config)
        self.latent_ffn_norm = nn.LayerNorm(config.d_model)
        self.latent_ffn = _FeedForward(config)

        self.asset_query_norm = nn.LayerNorm(config.d_model)
        self.latent_key_norm = nn.LayerNorm(config.d_model)
        self.asset_attention = _MultiHeadAttention(config)
        self.asset_ffn_norm = nn.LayerNorm(config.d_model)
        self.asset_ffn = _FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, assets: torch.Tensor, asset_valid: torch.Tensor) -> torch.Tensor:
        batch_size = assets.shape[0]
        panel_valid = asset_valid.any(dim=-1)
        latent_valid = panel_valid[:, None].expand(batch_size, self.latents.shape[0])
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        latents = latents * latent_valid.unsqueeze(-1)

        latents = latents + self.dropout(
            self.latent_attention(
                self.latent_query_norm(latents),
                self.asset_key_norm(assets),
                asset_valid,
                latent_valid,
            )
        )
        latents = latents * latent_valid.unsqueeze(-1)
        latents = latents + self.latent_ffn(self.latent_ffn_norm(latents))
        latents = latents * latent_valid.unsqueeze(-1)

        assets = assets + self.dropout(
            self.asset_attention(
                self.asset_query_norm(assets),
                self.latent_key_norm(latents),
                latent_valid,
                asset_valid,
            )
        )
        assets = assets * asset_valid.unsqueeze(-1)
        assets = assets + self.asset_ffn(self.asset_ffn_norm(assets))
        return assets * asset_valid.unsqueeze(-1)


class FactorPanelEncoder(nn.Module):
    """Encode a panel into per-asset features and a pooled factor embedding."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig.small()
        self.patch_projection = nn.Linear(
            self.config.patch_size * self.config.input_channels,
            self.config.d_model,
        )
        self.mask_token = nn.Parameter(torch.zeros(self.config.d_model))
        self.temporal_blocks = nn.ModuleList(
            _TemporalBlock(self.config) for _ in range(self.config.temporal_layers)
        )
        num_set_blocks = self.config.set_layers if self.config.use_set_mixer else 0
        self.set_blocks = nn.ModuleList(
            _InducedSetBlock(self.config) for _ in range(num_set_blocks)
        )
        self.pool_score = nn.Linear(self.config.d_model, 1, bias=False)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.output_dim),
        )

    def _validate_inputs(
        self,
        batch: FactorPanelBatch,
        views: InputViews | None,
        patch_mask: torch.Tensor | None,
    ) -> InputViews:
        if not isinstance(batch, FactorPanelBatch):
            raise TypeError("batch must be a FactorPanelBatch")
        if batch.context_length != self.config.context_length:
            raise ValueError(
                f"input context length {batch.context_length} does not match "
                f"configured context length {self.config.context_length}"
            )
        if batch.values.device != self.patch_projection.weight.device:
            raise ValueError("batch and model must be on the same device")
        if views is None:
            views = build_input_views(batch)
        elif not isinstance(views, InputViews):
            raise TypeError("views must be InputViews")
        if views.rank_gaussian.shape != batch.values.shape:
            raise ValueError("input views must match batch shape")
        if views.rank_gaussian.device != batch.values.device:
            raise ValueError("input views and batch must be on the same device")
        observed = views.observed_mask
        for numeric_view in (views.rank_gaussian, views.robust_z):
            finite_observed = torch.isfinite(numeric_view) | ~observed
            if not finite_observed.all().item():
                raise ValueError("observed numeric input views must be finite")
        if patch_mask is not None:
            expected = (batch.batch_size, batch.num_assets, self.config.num_patches)
            if not isinstance(patch_mask, torch.Tensor):
                raise TypeError("patch_mask must be a torch.Tensor")
            if patch_mask.dtype != torch.bool:
                raise TypeError("patch_mask must have bool dtype")
            if patch_mask.shape != expected:
                raise ValueError(f"patch_mask must have shape {expected}")
            if patch_mask.device != batch.values.device:
                raise ValueError("patch_mask and batch must be on the same device")
        return views

    def _patch_inputs(
        self,
        views: InputViews,
        patch_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = views.observed_mask
        channels = torch.stack(
            (
                torch.where(observed, views.rank_gaussian, 0.0),
                torch.where(observed, views.robust_z, 0.0),
                observed.to(dtype=views.rank_gaussian.dtype),
            ),
            dim=-1,
        ).permute(0, 2, 1, 3)
        if channels.shape[-1] != self.config.input_channels:
            raise ValueError(
                f"input views provide {channels.shape[-1]} channels, "
                f"expected {self.config.input_channels}"
            )
        patches = channels.unfold(
            2,
            self.config.patch_size,
            self.config.patch_stride,
        ).permute(0, 1, 2, 4, 3)
        patches = patches.flatten(-2).to(dtype=self.patch_projection.weight.dtype)

        observed = views.observed_mask.permute(0, 2, 1).unfold(
            2,
            self.config.patch_size,
            self.config.patch_stride,
        )
        patch_valid = observed.any(dim=-1)
        states = self.patch_projection(patches) * patch_valid.unsqueeze(-1)
        replace_mask = (
            torch.zeros_like(patch_valid) if patch_mask is None else patch_mask & patch_valid
        )
        states = torch.where(
            replace_mask.unsqueeze(-1),
            self.mask_token.view(1, 1, 1, -1),
            states,
        )
        return states, patch_valid

    @staticmethod
    def _last_valid(states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(states.shape[-2], device=states.device)
        indices = torch.where(valid, positions, -1).amax(dim=-1)
        gathered = states.gather(
            2,
            indices.clamp_min(0)[..., None, None].expand(-1, -1, 1, states.shape[-1]),
        ).squeeze(2)
        return gathered * (indices >= 0).unsqueeze(-1)

    def _pool(
        self,
        states: torch.Tensor,
        asset_valid: torch.Tensor,
    ) -> torch.Tensor:
        panel_valid = asset_valid.any(dim=-1)
        safe_valid = asset_valid.clone()
        safe_valid[:, 0] |= ~panel_valid
        scores = self.pool_score(states).squeeze(-1)
        scores = scores.masked_fill(~safe_valid, -torch.inf)
        weights = scores.softmax(dim=-1) * asset_valid
        pooled = torch.sum(weights.unsqueeze(-1) * states, dim=1)
        return pooled * panel_valid.unsqueeze(-1)

    def forward(
        self,
        batch: FactorPanelBatch,
        views: InputViews | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        views = self._validate_inputs(batch, views, patch_mask)
        patch_states, patch_valid = self._patch_inputs(views, patch_mask)
        batch_size, num_assets, num_patches, width = patch_states.shape

        flat_states = patch_states.reshape(batch_size * num_assets, num_patches, width)
        flat_valid = patch_valid.reshape(batch_size * num_assets, num_patches)
        for block in self.temporal_blocks:
            flat_states = block(flat_states, flat_valid)
        patch_states = flat_states.reshape(batch_size, num_assets, num_patches, width)

        temporal_states = self._last_valid(patch_states, patch_valid)
        asset_valid = patch_valid.any(dim=-1)
        contextual_states = temporal_states
        if self.config.use_set_mixer:
            for block in self.set_blocks:
                contextual_states = block(contextual_states, asset_valid)

        features = self.output_projection(contextual_states)
        features = features * asset_valid.unsqueeze(-1)
        factor_state = self._pool(contextual_states, asset_valid)
        factor_embedding = self.output_projection(factor_state)
        factor_embedding = factor_embedding * asset_valid.any(dim=-1, keepdim=True)
        return EncoderOutput(
            features=features,
            factor_embedding=factor_embedding,
            temporal_states=temporal_states,
            patch_states=patch_states,
            patch_valid=patch_valid,
            asset_valid=asset_valid,
        )

    def encode_factor(
        self,
        batch: FactorPanelBatch,
        views: InputViews | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Return the encoder output used by factor-model consumers."""

        return self.forward(batch, views=views, patch_mask=patch_mask)


__all__ = ["EncoderOutput", "FactorPanelEncoder", "ModelConfig"]
