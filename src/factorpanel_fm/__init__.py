"""FactorPanel-FM input contracts and Small encoder."""

from .batch import FactorPanelBatch
from .checkpoint import CheckpointInfo, load_checkpoint, save_checkpoint
from .data import (
    FactorPanelSample,
    PanelFrameDataset,
    chronological_split_indices,
    collate_factor_samples,
)
from .losses import (
    masked_huber_loss,
    negative_cross_sectional_ic_loss,
    pairwise_logistic_loss,
    quantile_pinball_loss,
)
from .model import EncoderOutput, FactorPanelEncoder, ModelConfig
from .runner import (
    RunSummary,
    RuntimeConfig,
    auto_micro_batch_probe,
    resolve_device,
    run_training,
    warmup_cosine_multiplier,
)
from .training import (
    StageAConfig,
    StageAModule,
    StageAOutput,
    StageBConfig,
    StageBModule,
    StageBOutput,
    build_stage_b_optimizer,
    configure_stage_b_trainability,
    sample_patch_mask,
    update_stage_b_optimizer,
)
from .views import InputViews, build_input_views

__all__ = [
    "CheckpointInfo",
    "EncoderOutput",
    "FactorPanelBatch",
    "FactorPanelSample",
    "FactorPanelEncoder",
    "InputViews",
    "ModelConfig",
    "PanelFrameDataset",
    "RunSummary",
    "RuntimeConfig",
    "StageAConfig",
    "StageAModule",
    "StageAOutput",
    "StageBConfig",
    "StageBModule",
    "StageBOutput",
    "build_stage_b_optimizer",
    "build_input_views",
    "auto_micro_batch_probe",
    "chronological_split_indices",
    "collate_factor_samples",
    "configure_stage_b_trainability",
    "load_checkpoint",
    "masked_huber_loss",
    "negative_cross_sectional_ic_loss",
    "pairwise_logistic_loss",
    "quantile_pinball_loss",
    "resolve_device",
    "run_training",
    "sample_patch_mask",
    "save_checkpoint",
    "update_stage_b_optimizer",
    "warmup_cosine_multiplier",
]
