from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import factorpanel_fm


class PublicApiTests(unittest.TestCase):
    def test_training_losses_and_checkpoint_symbols_are_exported(self) -> None:
        expected = {
            "CheckpointInfo",
            "StageAConfig",
            "StageAModule",
            "StageAOutput",
            "StageBConfig",
            "StageBModule",
            "StageBOutput",
            "build_stage_b_optimizer",
            "configure_stage_b_trainability",
            "load_checkpoint",
            "masked_huber_loss",
            "negative_cross_sectional_ic_loss",
            "pairwise_logistic_loss",
            "quantile_pinball_loss",
            "sample_patch_mask",
            "save_checkpoint",
        }

        self.assertTrue(expected.issubset(set(factorpanel_fm.__all__)))
        self.assertTrue(all(hasattr(factorpanel_fm, name) for name in expected))


if __name__ == "__main__":
    unittest.main()
