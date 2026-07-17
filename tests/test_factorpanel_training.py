from pathlib import Path
import math
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorpanel_fm import FactorPanelBatch, FactorPanelEncoder, ModelConfig, build_input_views
from factorpanel_fm.losses import (
    masked_huber_loss,
    negative_cross_sectional_ic_loss,
    pairwise_logistic_loss,
)
from factorpanel_fm.training import (
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


def make_batch(
    config: ModelConfig,
    batch_size: int = 2,
    num_assets: int = 5,
    missing: bool = True,
) -> FactorPanelBatch:
    values = torch.randn(batch_size, config.context_length, num_assets)
    observed = torch.ones_like(values, dtype=torch.bool)
    if missing:
        observed[0, :, -1] = False
        values[0, :, -1] = float("nan")
        observed[-1, 0, 0] = False
        values[-1, 0, 0] = float("nan")
    return FactorPanelBatch(
        values=values,
        observed_mask=observed,
        asset_ids=torch.arange(num_assets).expand(batch_size, -1),
        dates=torch.arange(config.context_length).expand(batch_size, -1),
    )


class StageAConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self) -> None:
        config = StageAConfig()
        self.assertEqual(config.mask_weight, 1.0)
        self.assertEqual(config.future_weight, 0.5)
        self.assertEqual(config.consistency_weight, 0.1)
        self.assertEqual(config.future_horizons, (5, 20))
        self.assertEqual(config.quantiles, (0.1, 0.5, 0.9))
        self.assertEqual(config.mask_ratio, 0.4)

        for parameters in (
            {"mask_ratio": 0.0},
            {"mask_ratio": 1.1},
            {"future_horizons": ()},
            {"future_horizons": (5, 5)},
            {"quantiles": (0.5, 0.1)},
            {"future_weight": -1.0},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises((TypeError, ValueError)):
                    StageAConfig(**parameters)

    def test_patch_mask_sampling_is_ratio_controlled_and_reproducible(self) -> None:
        valid = torch.ones(2, 5, 4, dtype=torch.bool)
        valid[0, -1].zero_()

        first = sample_patch_mask(
            valid,
            0.4,
            generator=torch.Generator().manual_seed(17),
        )
        second = sample_patch_mask(
            valid,
            0.4,
            generator=torch.Generator().manual_seed(17),
        )

        torch.testing.assert_close(first, second)
        self.assertFalse(first[~valid].any())
        self.assertLessEqual(abs(first.sum().item() / valid.sum().item() - 0.4), 1 / valid.sum().item())


class StageAModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8)
        self.model_config = ModelConfig.tiny(dropout=0.0)
        self.config = StageAConfig()
        self.module = StageAModule(FactorPanelEncoder(self.model_config), self.config)
        self.batch = make_batch(self.model_config)

    def test_shapes_weighted_total_and_mask_target_contract(self) -> None:
        future_targets = torch.randn(2, 5, 2)
        future_mask = torch.ones_like(future_targets, dtype=torch.bool)
        patch_mask = torch.zeros(2, 5, self.model_config.num_patches, dtype=torch.bool)
        patch_mask[:, :, 1:3] = True

        output = self.module(
            self.batch,
            future_targets,
            future_mask,
            patch_mask=patch_mask,
        )

        self.assertIsInstance(output, StageAOutput)
        self.assertEqual(output.future_quantiles.shape, (2, 5, 2, 3))
        self.assertEqual(
            output.patch_reconstruction.shape,
            (2, 5, self.model_config.num_patches, self.model_config.patch_size, 2),
        )
        self.assertEqual(output.patch_mask.shape, patch_mask.shape)
        expected_total = (
            output.mask_loss
            + 0.5 * output.future_factor_loss
            + 0.1 * output.consistency_loss
        )
        torch.testing.assert_close(output.total_loss, expected_total)
        self.assertEqual(output.consistency_loss.item(), 0.0)

        views = build_input_views(self.batch)
        target = torch.stack((views.rank_gaussian, views.robust_z), dim=-1)
        target = target.permute(0, 2, 1, 3).unfold(
            2,
            self.model_config.patch_size,
            self.model_config.patch_stride,
        ).permute(0, 1, 2, 4, 3)
        observed = views.observed_mask.permute(0, 2, 1).unfold(
            2,
            self.model_config.patch_size,
            self.model_config.patch_stride,
        )
        reconstruction_mask = output.patch_mask.unsqueeze(-1) & observed
        expected_mask_loss = masked_huber_loss(
            output.patch_reconstruction,
            target,
            reconstruction_mask.unsqueeze(-1).expand_as(output.patch_reconstruction),
        )
        torch.testing.assert_close(output.mask_loss, expected_mask_loss)

    def test_sampled_mask_is_deterministic_and_missing_inputs_are_safe(self) -> None:
        targets = torch.full((2, 5, 2), float("nan"))
        target_mask = torch.zeros_like(targets, dtype=torch.bool)
        first = self.module(
            self.batch,
            targets,
            target_mask,
            generator=torch.Generator().manual_seed(123),
        )
        second = self.module(
            self.batch,
            targets,
            target_mask,
            generator=torch.Generator().manual_seed(123),
        )

        torch.testing.assert_close(first.patch_mask, second.patch_mask)
        self.assertTrue(torch.isfinite(first.total_loss))
        self.assertEqual(first.future_factor_loss.item(), 0.0)
        first.total_loss.backward()

    def test_second_view_consistency_uses_only_overlap(self) -> None:
        second_batch = make_batch(self.model_config)
        targets = torch.randn(2, 5, 2)
        target_mask = torch.ones_like(targets, dtype=torch.bool)
        overlap = torch.zeros(2, 5, dtype=torch.bool)
        overlap[:, :3] = True

        output = self.module(
            self.batch,
            targets,
            target_mask,
            patch_mask=torch.zeros(2, 5, self.model_config.num_patches, dtype=torch.bool),
            second_batch=second_batch,
            overlap_mask=overlap,
        )

        self.assertTrue(torch.isfinite(output.consistency_loss))
        self.assertGreaterEqual(output.consistency_loss.item(), 0.0)


class StageBModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.model_config = ModelConfig.tiny(dropout=0.0)
        self.config = StageBConfig(initial_freeze_steps=5)
        self.module = StageBModule(FactorPanelEncoder(self.model_config), self.config)
        self.batch = make_batch(self.model_config)

    def test_defaults_and_validation(self) -> None:
        defaults = StageBConfig()
        self.assertEqual(defaults.ic_weight, 1.0)
        self.assertEqual(defaults.pairwise_weight, 0.5)
        self.assertEqual(defaults.huber_weight, 0.2)
        self.assertEqual(defaults.horizons, (1, 5, 20))
        self.assertEqual(defaults.initial_freeze_steps, 5000)
        self.assertEqual(defaults.base_lr, 3e-4)
        self.assertEqual(defaults.unfreeze_lr_scale, 0.1)
        for parameters in (
            {"initial_freeze_steps": -1},
            {"base_lr": 0.0},
            {"unfreeze_lr_scale": 0.0},
            {"horizons": (1, 1)},
            {"pairwise_weight": float("nan")},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises((TypeError, ValueError)):
                    StageBConfig(**parameters)

    def test_shapes_and_weighted_total(self) -> None:
        targets = torch.randn(2, 5, 3)
        mask = torch.ones_like(targets, dtype=torch.bool)

        output = self.module(
            self.batch,
            targets,
            mask,
            generator=torch.Generator().manual_seed(3),
        )

        self.assertIsInstance(output, StageBOutput)
        self.assertEqual(output.return_scores.shape, (2, 5, 3))
        expected_ic = negative_cross_sectional_ic_loss(output.return_scores, targets, mask)
        expected_pairwise = pairwise_logistic_loss(
            output.return_scores,
            targets,
            mask,
            generator=torch.Generator().manual_seed(3),
        )
        expected_huber = masked_huber_loss(output.return_scores, targets, mask)
        torch.testing.assert_close(output.ic_loss, expected_ic)
        torch.testing.assert_close(output.pairwise_loss, expected_pairwise)
        torch.testing.assert_close(output.huber_loss, expected_huber)
        torch.testing.assert_close(
            output.total_loss,
            expected_ic + 0.5 * expected_pairwise + 0.2 * expected_huber,
        )

    def test_freeze_and_optimizer_groups_have_expected_lr_without_duplicates(self) -> None:
        deep_config = ModelConfig.tiny(dropout=0.0, temporal_layers=8)
        module = StageBModule(
            FactorPanelEncoder(deep_config),
            StageBConfig(initial_freeze_steps=5),
        )
        lower_parameters = {
            id(parameter)
            for block in module.encoder.temporal_blocks[:4]
            for parameter in block.parameters()
        }

        configure_stage_b_trainability(module, step=0)
        self.assertTrue(lower_parameters)
        self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters() if id(parameter) in lower_parameters))
        frozen_optimizer = build_stage_b_optimizer(module, step=0)
        frozen_ids = [id(parameter) for group in frozen_optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(frozen_ids), len(set(frozen_ids)))
        self.assertTrue(lower_parameters.isdisjoint(frozen_ids))
        self.assertEqual({group["lr"] for group in frozen_optimizer.param_groups}, {3e-4})

        configure_stage_b_trainability(module, step=5)
        self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))
        optimizer = build_stage_b_optimizer(module, step=5)
        grouped = {
            id(parameter): group["lr"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(len(grouped), sum(len(group["params"]) for group in optimizer.param_groups))
        self.assertEqual(set(grouped), {id(parameter) for parameter in module.parameters()})
        self.assertTrue(all(
            math.isclose(grouped[parameter_id], 3e-5)
            for parameter_id in lower_parameters
        ))
        self.assertTrue(all(
            learning_rate == 3e-4
            for parameter_id, learning_rate in grouped.items()
            if parameter_id not in lower_parameters
        ))
        self.assertIn(0.0, {group["weight_decay"] for group in optimizer.param_groups})
        self.assertIn(0.05, {group["weight_decay"] for group in optimizer.param_groups})


if __name__ == "__main__":
    unittest.main()
