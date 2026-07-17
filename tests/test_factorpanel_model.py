from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorpanel_fm import (
    EncoderOutput,
    FactorPanelBatch,
    FactorPanelEncoder,
    ModelConfig,
    build_input_views,
)


class ModelConfigTests(unittest.TestCase):
    def test_small_defaults_and_num_patches(self) -> None:
        config = ModelConfig.small()

        self.assertEqual(config.context_length, 256)
        self.assertEqual(config.input_channels, 3)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.patch_stride, 8)
        self.assertEqual(config.d_model, 384)
        self.assertEqual(config.temporal_layers, 8)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.ffn_dim, 1536)
        self.assertEqual(config.num_latents, 32)
        self.assertEqual(config.set_layers, 2)
        self.assertEqual(config.output_dim, 128)
        self.assertEqual(config.dropout, 0.1)
        self.assertTrue(config.use_set_mixer)
        self.assertEqual(config.num_patches, 31)

    def test_rejects_invalid_dimensions_and_attention_geometry(self) -> None:
        invalid = (
            {"context_length": 0},
            {"input_channels": 0},
            {"patch_size": 0},
            {"patch_stride": 0},
            {"d_model": 0},
            {"temporal_layers": 0},
            {"num_heads": 0},
            {"ffn_dim": 0},
            {"num_latents": 0},
            {"set_layers": 0},
            {"output_dim": 0},
            {"context_length": 8, "patch_size": 16},
            {"d_model": 30, "num_heads": 8},
            {"d_model": 24, "num_heads": 8},
            {"dropout": -0.1},
            {"dropout": 1.0},
        )

        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    ModelConfig.tiny(**overrides)


class FactorPanelEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.config = ModelConfig.tiny(dropout=0.0)
        self.model = FactorPanelEncoder(self.config).eval()

    def make_batch(
        self,
        batch_size: int = 2,
        num_assets: int = 5,
        all_missing: bool = False,
    ) -> FactorPanelBatch:
        values = torch.randn(batch_size, self.config.context_length, num_assets)
        observed = torch.ones_like(values, dtype=torch.bool)
        if all_missing:
            values.fill_(float("nan"))
            observed.zero_()
        else:
            observed[0, :, -1] = False
            values[0, :, -1] = float("nan")
            observed[-1, : self.config.patch_size, 1] = False
            values[-1, : self.config.patch_size, 1] = float("nan")
        return FactorPanelBatch(
            values=values,
            observed_mask=observed,
            asset_ids=torch.arange(num_assets).expand(batch_size, -1),
            dates=torch.arange(self.config.context_length).expand(batch_size, -1),
        )

    def test_output_and_auxiliary_shapes_are_exact(self) -> None:
        batch = self.make_batch()

        output = self.model(batch)

        self.assertIsInstance(output, EncoderOutput)
        self.assertEqual(output.features.shape, (2, 5, self.config.output_dim))
        self.assertEqual(output.factor_embedding.shape, (2, self.config.output_dim))
        self.assertEqual(output.temporal_states.shape, (2, 5, self.config.d_model))
        self.assertEqual(
            output.patch_states.shape,
            (2, 5, self.config.num_patches, self.config.d_model),
        )
        self.assertEqual(output.patch_valid.shape, (2, 5, self.config.num_patches))
        self.assertEqual(output.asset_valid.shape, (2, 5))
        self.assertEqual(output.patch_valid.dtype, torch.bool)
        self.assertEqual(output.asset_valid.dtype, torch.bool)
        encoded = self.model.encode_factor(batch, views=build_input_views(batch))
        torch.testing.assert_close(encoded.features, output.features)
        torch.testing.assert_close(encoded.factor_embedding, output.factor_embedding)

    def test_all_missing_panels_are_finite_and_exactly_zero(self) -> None:
        output = self.model(self.make_batch(all_missing=True))

        for tensor in (
            output.features,
            output.factor_embedding,
            output.temporal_states,
            output.patch_states,
        ):
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertEqual(torch.count_nonzero(tensor).item(), 0)
        self.assertFalse(output.patch_valid.any())
        self.assertFalse(output.asset_valid.any())

    def test_asset_permutation_reorders_features_and_preserves_factor(self) -> None:
        batch = self.make_batch(batch_size=1, num_assets=6)
        permutation = torch.tensor([4, 1, 5, 0, 3, 2])
        permuted = FactorPanelBatch(
            values=batch.values[:, :, permutation],
            observed_mask=batch.observed_mask[:, :, permutation],
            asset_ids=batch.asset_ids[:, permutation],
            dates=batch.dates,
        )

        original_output = self.model(batch)
        permuted_output = self.model(permuted)

        torch.testing.assert_close(
            permuted_output.features,
            original_output.features[:, permutation],
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            permuted_output.factor_embedding,
            original_output.factor_embedding,
            atol=2e-5,
            rtol=2e-5,
        )

    def test_patch_mask_changes_valid_content_without_changing_validity(self) -> None:
        batch = self.make_batch(batch_size=1, num_assets=3)
        baseline = self.model(batch)
        patch_mask = torch.zeros_like(baseline.patch_valid)
        patch_mask[:, 0, 0] = True

        masked = self.model(batch, patch_mask=patch_mask)

        torch.testing.assert_close(masked.patch_valid, baseline.patch_valid)
        self.assertFalse(torch.allclose(masked.patch_states, baseline.patch_states))
        with self.assertRaises(ValueError):
            self.model(batch, patch_mask=torch.zeros(1, 3, 1, dtype=torch.bool))
        with self.assertRaises(TypeError):
            self.model(batch, patch_mask=patch_mask.float())

    def test_temporal_only_ablation_uses_same_output_contract(self) -> None:
        config = ModelConfig.tiny(dropout=0.0, use_set_mixer=False)
        model = FactorPanelEncoder(config).eval()

        output = model(self.make_batch())

        self.assertEqual(output.features.shape, (2, 5, config.output_dim))
        self.assertEqual(output.factor_embedding.shape, (2, config.output_dim))
        self.assertTrue(torch.isfinite(output.features).all())
        self.assertEqual(torch.count_nonzero(output.features[0, -1]).item(), 0)

    def test_rejects_input_context_mismatch(self) -> None:
        length = self.config.context_length - 1
        values = torch.randn(1, length, 2)
        batch = FactorPanelBatch(
            values=values,
            observed_mask=torch.ones_like(values, dtype=torch.bool),
            asset_ids=torch.arange(2).unsqueeze(0),
            dates=torch.arange(length).unsqueeze(0),
        )

        with self.assertRaisesRegex(ValueError, "context"):
            self.model(batch)

    def test_small_parameter_count_is_in_target_band(self) -> None:
        parameter_count = sum(
            parameter.numel()
            for parameter in FactorPanelEncoder(ModelConfig.small()).parameters()
        )

        self.assertGreaterEqual(parameter_count, 20_000_000)
        self.assertLessEqual(parameter_count, 26_000_000)


if __name__ == "__main__":
    unittest.main()
