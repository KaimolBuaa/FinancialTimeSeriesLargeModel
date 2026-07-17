from pathlib import Path
import math
import tempfile
import sys
import unittest

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F


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
    update_stage_b_optimizer,
    sample_patch_mask,
    _randperm_for_device,
    _expand_patch_mask_for_overlap,
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


def _stage_b_freeze_ddp_worker(rank: int, world_size: int, store_path: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(101 + rank)
        config = ModelConfig.tiny(dropout=0.0, temporal_layers=4)
        module = StageBModule(
            FactorPanelEncoder(config),
            StageBConfig(
                initial_freeze_steps=2,
                base_lr=1e-2,
                ic_weight=0.0,
                pairwise_weight=0.0,
                huber_weight=1.0,
            ),
        )
        optimizer = build_stage_b_optimizer(module, step=0)
        if not all(parameter.requires_grad for parameter in module.parameters()):
            raise AssertionError("Stage B parameters must stay in the DDP graph while frozen")
        distributed = DistributedDataParallel(module)
        batch = make_batch(config, batch_size=1, num_assets=4, missing=False)
        lower = [
            parameter
            for block in module.encoder.temporal_blocks[:2]
            for parameter in block.parameters()
        ]
        initial = [parameter.detach().clone() for parameter in lower]

        for step in range(2):
            update_stage_b_optimizer(module, optimizer, step)
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(1000 * rank + step)
            output = distributed(
                batch,
                torch.randn(1, 4, 3),
                torch.ones(1, 4, 3, dtype=torch.bool),
            )
            output.total_loss.backward()
            if any(parameter.grad is None for parameter in lower):
                raise AssertionError("frozen lower parameters must receive synchronized gradients")
            flattened = torch.cat([parameter.grad.flatten() for parameter in lower])
            gathered = [torch.empty_like(flattened) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, flattened)
            if not all(torch.allclose(gathered[0], other) for other in gathered[1:]):
                raise AssertionError("lower gradients differ across DDP ranks")
            optimizer.step()
            if any(not torch.equal(parameter, expected) for parameter, expected in zip(lower, initial)):
                raise AssertionError("zero-lr lower parameters changed during the freeze period")

        update_stage_b_optimizer(module, optimizer, step=2)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(2000 + rank)
        output = distributed(
            batch,
            torch.randn(1, 4, 3),
            torch.ones(1, 4, 3, dtype=torch.bool),
        )
        output.total_loss.backward()
        flattened = torch.cat([parameter.grad.flatten() for parameter in lower])
        gathered = [torch.empty_like(flattened) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, flattened)
        if not all(torch.allclose(gathered[0], other) for other in gathered[1:]):
            raise AssertionError("unfrozen lower gradients differ across DDP ranks")
        optimizer.step()
        if not any(not torch.equal(parameter, expected) for parameter, expected in zip(lower, initial)):
            raise AssertionError("lower parameters did not update after the freeze boundary")
    finally:
        torch.distributed.destroy_process_group()


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

    def test_cpu_generator_random_helper_matches_cpu_reference(self) -> None:
        expected = torch.randperm(19, generator=torch.Generator().manual_seed(41))
        actual = _randperm_for_device(
            19,
            torch.device("cpu"),
            generator=torch.Generator().manual_seed(41),
        )
        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_cpu_generator_can_sample_mps_patch_mask(self) -> None:
        valid = torch.ones(2, 5, 4, dtype=torch.bool, device="mps")

        sampled = sample_patch_mask(
            valid,
            0.4,
            generator=torch.Generator().manual_seed(17),
        )

        self.assertEqual(sampled.device.type, "mps")
        self.assertEqual(sampled.sum().cpu().item(), 16)

    def test_patch_overlap_expansion_uses_exact_interval_geometry(self) -> None:
        selected = torch.zeros(1, 1, 6, dtype=torch.bool)
        selected[..., 0] = True
        torch.testing.assert_close(
            _expand_patch_mask_for_overlap(selected, patch_size=5, patch_stride=2),
            torch.tensor([[[True, True, True, False, False, False]]]),
        )

        selected.zero_()
        selected[..., 3] = True
        torch.testing.assert_close(
            _expand_patch_mask_for_overlap(selected, patch_size=5, patch_stride=2),
            torch.tensor([[[False, True, True, True, True, True]]]),
        )
        torch.testing.assert_close(
            _expand_patch_mask_for_overlap(selected, patch_size=4, patch_stride=4),
            selected,
        )


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
        self.assertEqual(output.encoder_patch_mask.shape, patch_mask.shape)
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

    def test_encoder_mask_expands_around_reconstruction_targets(self) -> None:
        config = ModelConfig.tiny(
            context_length=13,
            patch_size=5,
            patch_stride=2,
            dropout=0.0,
        )
        module = StageAModule(FactorPanelEncoder(config))
        batch = make_batch(config, batch_size=1, num_assets=2, missing=False)
        selected = torch.zeros(1, 2, config.num_patches, dtype=torch.bool)
        selected[:, :, 0] = True

        output = module(
            batch,
            torch.randn(1, 2, 2),
            torch.ones(1, 2, 2, dtype=torch.bool),
            patch_mask=selected,
        )

        torch.testing.assert_close(output.patch_mask, selected)
        expected = torch.zeros_like(selected)
        expected[:, :, :3] = True
        torch.testing.assert_close(output.encoder_patch_mask, expected)

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

    def test_second_view_matches_reordered_assets_by_id(self) -> None:
        permutation = torch.tensor([3, 0, 4, 1, 2])
        second_batch = FactorPanelBatch(
            values=self.batch.values[:, :, permutation],
            observed_mask=self.batch.observed_mask[:, :, permutation],
            asset_ids=self.batch.asset_ids[:, permutation],
            dates=self.batch.dates,
        )

        output = self.module(
            self.batch,
            torch.randn(2, 5, 2),
            torch.ones(2, 5, 2, dtype=torch.bool),
            patch_mask=torch.zeros(2, 5, self.model_config.num_patches, dtype=torch.bool),
            second_batch=second_batch,
        )

        torch.testing.assert_close(output.consistency_loss, torch.tensor(0.0), atol=1e-6, rtol=0)

    def test_second_view_accepts_asset_subsample_and_first_view_overlap_mask(self) -> None:
        selected = torch.tensor([4, 1, 3])
        second_batch = FactorPanelBatch(
            values=self.batch.values[:, :, selected],
            observed_mask=self.batch.observed_mask[:, :, selected],
            asset_ids=self.batch.asset_ids[:, selected],
            dates=self.batch.dates,
        )
        overlap_mask = torch.zeros(2, 5, dtype=torch.bool)
        overlap_mask[:, selected] = True

        output = self.module(
            self.batch,
            torch.randn(2, 5, 2),
            torch.ones(2, 5, 2, dtype=torch.bool),
            patch_mask=torch.zeros(2, 5, self.model_config.num_patches, dtype=torch.bool),
            second_batch=second_batch,
            overlap_mask=overlap_mask,
        )
        second_output = self.module.encoder(second_batch)
        valid = output.encoder_output.asset_valid[:, selected] & second_output.asset_valid
        expected = 1.0 - F.cosine_similarity(
            output.encoder_output.features[:, selected][valid],
            second_output.features[valid],
            dim=-1,
        ).mean()
        torch.testing.assert_close(output.consistency_loss, expected)

    def test_second_view_rejects_mismatched_dates_and_duplicate_asset_ids(self) -> None:
        targets = torch.randn(2, 5, 2)
        target_mask = torch.ones_like(targets, dtype=torch.bool)
        bad_dates = FactorPanelBatch(
            values=self.batch.values,
            observed_mask=self.batch.observed_mask,
            asset_ids=self.batch.asset_ids,
            dates=self.batch.dates + 1,
        )
        with self.assertRaisesRegex(ValueError, "dates"):
            self.module(self.batch, targets, target_mask, second_batch=bad_dates)

        duplicate_ids = self.batch.asset_ids.clone()
        duplicate_ids[:, 1] = duplicate_ids[:, 0]
        duplicate_second = FactorPanelBatch(
            values=self.batch.values,
            observed_mask=self.batch.observed_mask,
            asset_ids=duplicate_ids,
            dates=self.batch.dates,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            self.module(self.batch, targets, target_mask, second_batch=duplicate_second)

        duplicate_first = FactorPanelBatch(
            values=self.batch.values,
            observed_mask=self.batch.observed_mask,
            asset_ids=duplicate_ids,
            dates=self.batch.dates,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            self.module(duplicate_first, targets, target_mask, second_batch=self.batch)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_second_view_supports_mps_features_with_cpu_coordinates(self) -> None:
        module = StageAModule(FactorPanelEncoder(self.model_config)).to("mps")
        values = torch.randn(1, self.model_config.context_length, 4, device="mps")
        asset_ids = torch.tensor([[10, 20, 30, 40]])
        dates = torch.arange(self.model_config.context_length).unsqueeze(0)
        batch = FactorPanelBatch(
            values=values,
            observed_mask=torch.ones_like(values, dtype=torch.bool),
            asset_ids=asset_ids,
            dates=dates,
        )
        permutation = torch.tensor([2, 0, 3], device="mps")
        second_batch = FactorPanelBatch(
            values=values.index_select(2, permutation),
            observed_mask=torch.ones(1, self.model_config.context_length, 3, dtype=torch.bool, device="mps"),
            asset_ids=asset_ids[:, [2, 0, 3]],
            dates=dates,
        )

        output = module(
            batch,
            torch.randn(1, 4, 2, device="mps"),
            torch.ones(1, 4, 2, dtype=torch.bool, device="mps"),
            patch_mask=torch.zeros(1, 4, self.model_config.num_patches, dtype=torch.bool, device="mps"),
            second_batch=second_batch,
        )

        self.assertEqual(output.consistency_loss.device.type, "mps")
        self.assertTrue(torch.isfinite(output.consistency_loss).cpu().item())


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

    def test_total_loss_backward_preserves_mask_and_reaches_encoder(self) -> None:
        targets = torch.randn(2, 5, 3)
        mask = torch.rand(2, 5, 3) > 0.2
        expected_mask = mask.clone()

        output = self.module(
            self.batch,
            targets,
            mask,
            generator=torch.Generator().manual_seed(5),
        )
        output.total_loss.backward()

        torch.testing.assert_close(mask, expected_mask)
        missing = [
            name
            for name, parameter in self.module.encoder.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])

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
        self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))
        frozen_optimizer = build_stage_b_optimizer(module, step=0)
        frozen_ids = [id(parameter) for group in frozen_optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(frozen_ids), len(set(frozen_ids)))
        self.assertEqual(set(frozen_ids), {id(parameter) for parameter in module.parameters()})
        frozen_grouped = {
            id(parameter): group["lr"]
            for group in frozen_optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(all(frozen_grouped[parameter_id] == 0.0 for parameter_id in lower_parameters))
        self.assertTrue(all(
            learning_rate == 3e-4
            for parameter_id, learning_rate in frozen_grouped.items()
            if parameter_id not in lower_parameters
        ))

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

    def test_optimizer_updates_in_place_and_preserves_existing_adam_state(self) -> None:
        module = StageBModule(
            FactorPanelEncoder(ModelConfig.tiny(dropout=0.0, temporal_layers=4)),
            StageBConfig(initial_freeze_steps=2),
        )
        batch = make_batch(module.encoder.config, batch_size=1, num_assets=4)
        optimizer = build_stage_b_optimizer(module, step=0)
        group_parameters = [tuple(map(id, group["params"])) for group in optimizer.param_groups]
        output = module(
            batch,
            torch.randn(1, 4, 3),
            torch.ones(1, 4, 3, dtype=torch.bool),
        )
        output.total_loss.backward()
        optimizer.step()
        upper_parameter = module.return_head.weight
        expected_momentum = optimizer.state[upper_parameter]["exp_avg"].clone()

        result = update_stage_b_optimizer(module, optimizer, step=2)

        self.assertIs(result, optimizer)
        self.assertEqual(
            [tuple(map(id, group["params"])) for group in optimizer.param_groups],
            group_parameters,
        )
        torch.testing.assert_close(optimizer.state[upper_parameter]["exp_avg"], expected_momentum)
        self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))

    def test_optimizer_state_dict_restores_across_freeze_boundary(self) -> None:
        config = ModelConfig.tiny(dropout=0.0, temporal_layers=4)
        frozen_module = StageBModule(
            FactorPanelEncoder(config),
            StageBConfig(initial_freeze_steps=2),
        )
        unfrozen_module = StageBModule(
            FactorPanelEncoder(config),
            StageBConfig(initial_freeze_steps=2),
        )
        frozen_optimizer = build_stage_b_optimizer(frozen_module, step=0)
        unfrozen_optimizer = build_stage_b_optimizer(unfrozen_module, step=2)

        unfrozen_optimizer.load_state_dict(frozen_optimizer.state_dict())
        update_stage_b_optimizer(unfrozen_module, unfrozen_optimizer, step=2)

        self.assertEqual(
            [len(group["params"]) for group in unfrozen_optimizer.param_groups],
            [len(group["params"]) for group in frozen_optimizer.param_groups],
        )
        lower_lrs = {
            group["lr"]
            for group in unfrozen_optimizer.param_groups
            if group["stage_b_lower"]
        }
        self.assertTrue(all(math.isclose(learning_rate, 3e-5) for learning_rate in lower_lrs))


class DistributedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._store_directory = tempfile.TemporaryDirectory()
        store_path = Path(cls._store_directory.name) / "ddp-store"
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{store_path}",
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        torch.distributed.destroy_process_group()
        cls._store_directory.cleanup()

    def test_stage_a_and_b_support_two_default_ddp_iterations(self) -> None:
        config = ModelConfig.tiny(dropout=0.0)
        batch = make_batch(config, batch_size=1, num_assets=4)
        patch_mask = torch.zeros(1, 4, config.num_patches, dtype=torch.bool)
        patch_mask[:, :, 1] = True

        stage_a = DistributedDataParallel(StageAModule(FactorPanelEncoder(config)))
        for _ in range(2):
            stage_a.zero_grad(set_to_none=True)
            output_a = stage_a(
                batch,
                torch.randn(1, 4, 2),
                torch.ones(1, 4, 2, dtype=torch.bool),
                patch_mask=patch_mask,
            )
            output_a.total_loss.backward()
            self.assertEqual(
                [
                    name
                    for name, parameter in stage_a.module.encoder.named_parameters()
                    if parameter.grad is None
                ],
                [],
            )

        stage_b = DistributedDataParallel(
            StageBModule(FactorPanelEncoder(config), StageBConfig(initial_freeze_steps=0))
        )
        for _ in range(2):
            stage_b.zero_grad(set_to_none=True)
            output_b = stage_b(
                batch,
                torch.randn(1, 4, 3),
                torch.ones(1, 4, 3, dtype=torch.bool),
            )
            output_b.total_loss.backward()
            self.assertEqual(
                [
                    name
                    for name, parameter in stage_b.module.encoder.named_parameters()
                    if parameter.grad is None
                ],
                [],
            )


class MultiProcessDistributedTrainingTests(unittest.TestCase):
    def test_stage_b_freeze_boundary_keeps_lower_gradients_in_two_rank_ddp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = str(Path(directory) / "stage-b-ddp-store")
            torch.multiprocessing.spawn(
                _stage_b_freeze_ddp_worker,
                args=(2, store_path),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
