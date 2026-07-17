from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorpanel_fm import FactorPanelBatch, InputViews, build_input_views


class FactorPanelBatchTests(unittest.TestCase):
    def make_batch(self) -> FactorPanelBatch:
        values = torch.tensor(
            [[[1.0, 2.0, float("nan")], [3.0, 4.0, 5.0]]],
            dtype=torch.float32,
        )
        observed_mask = torch.tensor(
            [[[True, True, False], [True, True, True]]],
            dtype=torch.bool,
        )
        return FactorPanelBatch(
            values=values,
            observed_mask=observed_mask,
            asset_ids=torch.tensor([[10, 20, 30]], dtype=torch.int64),
            dates=torch.tensor([[20200101, 20200102]], dtype=torch.int64),
        )

    def test_properties_report_panel_dimensions(self) -> None:
        batch = self.make_batch()

        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch.context_length, 2)
        self.assertEqual(batch.num_assets, 3)

    def test_to_returns_a_new_batch_on_requested_device(self) -> None:
        batch = self.make_batch()

        moved = batch.to("cpu")

        self.assertIsNot(moved, batch)
        self.assertEqual(moved.values.device.type, "cpu")
        self.assertEqual(moved.observed_mask.device.type, "cpu")
        self.assertEqual(moved.asset_ids.device.type, "cpu")
        self.assertEqual(moved.dates.device.type, "cpu")

    def test_rejects_invalid_ranks_shapes_and_empty_dimensions(self) -> None:
        valid = self.make_batch()
        invalid_cases = (
            {"values": torch.ones(2, 3)},
            {"observed_mask": torch.ones(1, 2, 2, dtype=torch.bool)},
            {"asset_ids": torch.ones(1, 2, dtype=torch.int64)},
            {"dates": torch.ones(1, 3, dtype=torch.int64)},
            {
                "values": torch.empty(1, 0, 3),
                "observed_mask": torch.empty(1, 0, 3, dtype=torch.bool),
                "dates": torch.empty(1, 0, dtype=torch.int64),
            },
        )
        fields = {
            "values": valid.values,
            "observed_mask": valid.observed_mask,
            "asset_ids": valid.asset_ids,
            "dates": valid.dates,
        }

        for overrides in invalid_cases:
            with self.subTest(overrides=tuple(overrides)):
                with self.assertRaises(ValueError):
                    FactorPanelBatch(**(fields | overrides))

    def test_rejects_incorrect_dtypes(self) -> None:
        valid = self.make_batch()
        invalid_cases = (
            {"values": valid.values.to(torch.int64)},
            {"observed_mask": valid.observed_mask.to(torch.float32)},
            {"asset_ids": valid.asset_ids.to(torch.float32)},
            {"dates": valid.dates.to(torch.float32)},
        )
        fields = {
            "values": valid.values,
            "observed_mask": valid.observed_mask,
            "asset_ids": valid.asset_ids,
            "dates": valid.dates,
        }

        for overrides in invalid_cases:
            with self.subTest(overrides=tuple(overrides)):
                with self.assertRaises((TypeError, ValueError)):
                    FactorPanelBatch(**(fields | overrides))

    def test_rejects_nonfinite_observed_values_only(self) -> None:
        values = torch.tensor([[[1.0, float("inf")]]])
        common = {
            "values": values,
            "asset_ids": torch.tensor([[1, 2]]),
            "dates": torch.tensor([[20200101]]),
        }

        FactorPanelBatch(
            **common,
            observed_mask=torch.tensor([[[True, False]]]),
        )
        with self.assertRaises(ValueError):
            FactorPanelBatch(
                **common,
                observed_mask=torch.tensor([[[True, True]]]),
            )


class InputViewTests(unittest.TestCase):
    @staticmethod
    def make_batch(values: torch.Tensor, mask: torch.Tensor | None = None) -> FactorPanelBatch:
        batch_size, context_length, num_assets = values.shape
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        return FactorPanelBatch(
            values=values,
            observed_mask=mask,
            asset_ids=torch.arange(num_assets).expand(batch_size, -1),
            dates=torch.arange(context_length).expand(batch_size, -1),
        )

    def test_views_have_exact_shapes_and_stack_three_float_channels(self) -> None:
        batch = self.make_batch(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))

        views = build_input_views(batch)

        self.assertIsInstance(views, InputViews)
        self.assertEqual(views.rank_gaussian.shape, (2, 3, 4))
        self.assertEqual(views.robust_z.shape, (2, 3, 4))
        self.assertEqual(views.observed_mask.shape, (2, 3, 4))
        self.assertEqual(views.observed_mask.dtype, torch.bool)
        self.assertEqual(views.stacked.shape, (2, 3, 4, 3))
        self.assertTrue(views.stacked.dtype.is_floating_point)
        torch.testing.assert_close(views.stacked[..., 0], views.rank_gaussian)
        torch.testing.assert_close(views.stacked[..., 1], views.robust_z)
        torch.testing.assert_close(views.stacked[..., 2], views.observed_mask.float())

    def test_missing_entries_are_zero_in_numeric_views(self) -> None:
        values = torch.tensor([[[1.0, float("nan"), 3.0], [2.0, 4.0, 6.0]]])
        mask = torch.tensor([[[True, False, True], [False, True, True]]])

        views = build_input_views(self.make_batch(values, mask))

        self.assertEqual(views.rank_gaussian[0, 0, 1].item(), 0.0)
        self.assertEqual(views.rank_gaussian[0, 1, 0].item(), 0.0)
        self.assertEqual(views.robust_z[0, 0, 1].item(), 0.0)
        self.assertEqual(views.robust_z[0, 1, 0].item(), 0.0)
        self.assertFalse(views.observed_mask[0, 0, 1].item())

    def test_rank_gaussian_uses_midrank_percentiles(self) -> None:
        values = torch.tensor([[[1.0, 2.0, 3.0]]])

        views = build_input_views(self.make_batch(values))

        expected = torch.tensor([[[-0.9674216, 0.0, 0.9674216]]])
        torch.testing.assert_close(views.rank_gaussian, expected, atol=1e-6, rtol=1e-6)

    def test_tied_ranks_are_asset_permutation_equivariant(self) -> None:
        values = torch.tensor([[[2.0, 1.0, 2.0, 4.0], [3.0, 3.0, 1.0, 2.0]]])
        permutation = torch.tensor([3, 0, 2, 1])
        original = build_input_views(self.make_batch(values))
        permuted = build_input_views(self.make_batch(values[:, :, permutation]))

        torch.testing.assert_close(
            permuted.rank_gaussian,
            original.rank_gaussian[:, :, permutation],
        )
        self.assertEqual(original.rank_gaussian[0, 0, 0], original.rank_gaussian[0, 0, 2])

    def test_robust_z_uses_only_prior_observed_variable_history(self) -> None:
        values = torch.tensor([[[1.0], [2.0], [100.0], [4.0]]])

        views = build_input_views(self.make_batch(values), z_clip=200.0)

        expected = torch.tensor([[[0.0], [0.0], [98.5 / 0.7413], [2.0 / 1.4826]]])
        torch.testing.assert_close(views.robust_z, expected, atol=1e-4, rtol=1e-4)

    def test_robust_window_limits_history(self) -> None:
        values = torch.tensor([[[0.0], [10.0], [20.0], [15.0]]])

        views = build_input_views(self.make_batch(values), robust_window=2, z_clip=100.0)

        self.assertAlmostEqual(views.robust_z[0, 3, 0].item(), 0.0, places=6)

    def test_changing_future_values_does_not_change_earlier_views(self) -> None:
        values = torch.tensor(
            [[[1.0, 4.0, 2.0], [2.0, 5.0, 3.0], [3.0, 6.0, 4.0], [4.0, 7.0, 5.0]]]
        )
        changed = values.clone()
        changed[:, 2:] = torch.tensor([[[1000.0, -1000.0, 500.0], [-9.0, 9.0, 0.0]]])

        original_views = build_input_views(self.make_batch(values))
        changed_views = build_input_views(self.make_batch(changed))

        torch.testing.assert_close(
            changed_views.stacked[:, :2],
            original_views.stacked[:, :2],
        )


if __name__ == "__main__":
    unittest.main()
