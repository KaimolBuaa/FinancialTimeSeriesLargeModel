"""FactorPanel-FM input contracts and Small encoder."""

from .batch import FactorPanelBatch
from .checkpoint import CheckpointInfo, load_checkpoint, save_checkpoint
from .losses import (
    masked_huber_loss,
    negative_cross_sectional_ic_loss,
    pairwise_logistic_loss,
    quantile_pinball_loss,
)
from .model import EncoderOutput, FactorPanelEncoder, ModelConfig
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
)
from .views import InputViews, build_input_views

__all__ = [
    "CheckpointInfo",
    "EncoderOutput",
    "FactorPanelBatch",
    "FactorPanelEncoder",
    "InputViews",
    "ModelConfig",
    "StageAConfig",
    "StageAModule",
    "StageAOutput",
    "StageBConfig",
    "StageBModule",
    "StageBOutput",
    "build_stage_b_optimizer",
    "build_input_views",
    "configure_stage_b_trainability",
    "load_checkpoint",
    "masked_huber_loss",
    "negative_cross_sectional_ic_loss",
    "pairwise_logistic_loss",
    "quantile_pinball_loss",
    "sample_patch_mask",
    "save_checkpoint",
]
