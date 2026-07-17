from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorpanel_fm import (  # noqa: E402
    FactorPanelBatch,
    FactorPanelSample,
    PanelFrameDataset,
    chronological_split_indices,
    collate_factor_samples,
)


class FactorPanelSampleTests(unittest.TestCase):
    @staticmethod
    def make_sample(
        factor_id: str = "value", decision_date: int = 3
    ) -> FactorPanelSample:
        batch = FactorPanelBatch(
            values=torch.arange(6, dtype=torch.float32).reshape(1, 3, 2),
            observed_mask=torch.ones(1, 3, 2, dtype=torch.bool),
            asset_ids=torch.tensor([[10, 20]], dtype=torch.int64),
            dates=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        )
        return FactorPanelSample(
            factor_id=factor_id,
            batch=batch,
            future_factor_targets=torch.ones(1, 2, 2),
            future_factor_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            return_targets=torch.ones(1, 2, 3),
            return_mask=torch.ones(1, 2, 3, dtype=torch.bool),
            decision_date=decision_date,
        )

    def test_rejects_bad_shapes_dtypes_and_devices(self) -> None:
        sample = self.make_sample()
        fields = {
            "factor_id": sample.factor_id,
            "batch": sample.batch,
            "future_factor_targets": sample.future_factor_targets,
            "future_factor_mask": sample.future_factor_mask,
            "return_targets": sample.return_targets,
            "return_mask": sample.return_mask,
            "decision_date": sample.decision_date,
        }
        bad = (
            {"future_factor_targets": torch.ones(1, 3, 2)},
            {"future_factor_mask": torch.ones(1, 2, 2)},
            {"return_targets": torch.ones(1, 2, 3, dtype=torch.int64)},
            {"return_mask": torch.ones(1, 2, 2, dtype=torch.bool)},
            {"future_factor_targets": torch.ones(1, 2, 2, device="meta")},
            {"decision_date": True},
        )
        for overrides in bad:
            with self.subTest(overrides=tuple(overrides)):
                with self.assertRaises((TypeError, ValueError)):
                    FactorPanelSample(**(fields | overrides))

    def test_collate_stacks_singletons_and_preserves_coordinates(self) -> None:
        first = self.make_sample("alpha", 3)
        second = self.make_sample("beta", 4)

        collated = collate_factor_samples([first, second])

        self.assertEqual(collated.factor_id, ("alpha", "beta"))
        self.assertEqual(collated.decision_date, (3, 4))
        self.assertEqual(collated.batch.values.shape, (2, 3, 2))
        self.assertEqual(collated.future_factor_targets.shape, (2, 2, 2))
        self.assertEqual(collated.return_targets.shape, (2, 2, 3))

        incompatible = self.make_sample()
        object.__setattr__(
            incompatible,
            "batch",
            FactorPanelBatch(
                values=torch.ones(1, 3, 2),
                observed_mask=torch.ones(1, 3, 2, dtype=torch.bool),
                asset_ids=torch.tensor([[10, 99]]),
                dates=torch.tensor([[1, 2, 3]]),
            ),
        )
        with self.assertRaisesRegex(ValueError, "asset_ids"):
            collate_factor_samples([first, incompatible])


class PanelFrameDatasetTests(unittest.TestCase):
    @staticmethod
    def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        labels = []
        for date in range(1, 8):
            for asset_index, asset in enumerate(("B", "A")):
                rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "value": date * 10 + asset_index,
                        "quality": date * 100 + asset_index,
                    }
                )
                labels.append(
                    {
                        "date": date,
                        "asset": asset,
                        "r1": date + asset_index / 10,
                        "r5": date + asset_index / 5,
                    }
                )
        factors = pd.DataFrame(rows)
        returns = pd.DataFrame(labels)
        factors.loc[(factors.date == 2) & (factors.asset == "A"), "value"] = float(
            "nan"
        )
        factors = factors.drop(
            factors[(factors.date == 3) & (factors.asset == "B")].index
        )
        returns = returns.drop(
            returns[(returns.date == 3) & (returns.asset == "A")].index
        )
        return factors, returns

    def test_materializes_factor_windows_future_states_and_current_returns(
        self,
    ) -> None:
        factors, labels = self.frames()
        dataset = PanelFrameDataset(
            factors,
            labels,
            context_length=3,
            future_horizons=(1, 2),
            stride=2,
            factor_columns=("value", "quality"),
            return_columns=("r1", "r5"),
        )

        self.assertEqual(len(dataset), 4)
        sample = dataset[0]
        self.assertEqual(sample.factor_id, "value")
        self.assertEqual(sample.decision_date, 3)
        self.assertEqual(sample.batch.asset_ids.tolist(), [[0, 1]])
        self.assertEqual(sample.batch.dates.tolist(), [[1, 2, 3]])
        self.assertEqual(sample.batch.values.shape, (1, 3, 2))
        self.assertFalse(sample.batch.observed_mask[0, 1, 0].item())
        self.assertFalse(sample.batch.observed_mask[0, 2, 1].item())
        torch.testing.assert_close(
            sample.future_factor_targets[0],
            torch.tensor([[41.0, 51.0], [40.0, 50.0]]),
        )
        torch.testing.assert_close(
            sample.return_targets[0, 0],
            torch.tensor([0.0, 0.0]),
        )
        self.assertFalse(sample.return_mask[0, 0].any())
        self.assertTrue(sample.return_mask[0, 1].all())
        self.assertEqual(dataset[2].factor_id, "quality")

    def test_duplicate_keys_are_rejected_and_short_panel_is_empty(self) -> None:
        factors, labels = self.frames()
        duplicate = pd.concat([factors, factors.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            PanelFrameDataset(duplicate, context_length=2, future_horizons=(1,))

        short = PanelFrameDataset(
            factors[factors.date <= 3],
            context_length=3,
            future_horizons=(2,),
        )
        self.assertEqual(len(short), 0)

    def test_empty_future_horizons_keep_all_stage_b_decision_dates(self) -> None:
        factors = pd.DataFrame(
            {
                "date": [date for date in range(1, 37) for _ in range(2)],
                "asset": [asset for _ in range(1, 37) for asset in ("A", "B")],
                "value": [float(date) for date in range(1, 37) for _ in range(2)],
            }
        )
        dataset = PanelFrameDataset(
            factors,
            context_length=16,
            future_horizons=(),
            factor_columns=("value",),
        )

        self.assertEqual(len(dataset), 21)
        self.assertEqual(dataset[-1].decision_date, 36)
        self.assertEqual(dataset[-1].future_factor_targets.shape, (1, 2, 0))
        self.assertEqual(dataset[-1].future_factor_mask.shape, (1, 2, 0))

    def test_from_parquet_matches_frame_adapter(self) -> None:
        factors, labels = self.frames()
        with tempfile.TemporaryDirectory() as directory:
            factor_path = Path(directory) / "factors.parquet"
            label_path = Path(directory) / "labels.parquet"
            factors.to_parquet(factor_path)
            labels.to_parquet(label_path)
            dataset = PanelFrameDataset.from_parquet(
                factor_path,
                label_path,
                context_length=3,
                future_horizons=(1, 2),
                factor_columns=("value",),
                return_columns=("r1", "r5"),
            )

        self.assertEqual(dataset[0].decision_date, 3)
        self.assertEqual(dataset[0].factor_id, "value")

    def test_chronological_split_uses_date_positions_for_purge(self) -> None:
        factors = pd.DataFrame(
            {
                "date": [10, 20, 40, 70, 110, 160, 220, 300],
                "asset": ["A"] * 8,
                "value": list(range(8)),
            }
        )
        dataset = PanelFrameDataset(
            factors,
            context_length=1,
            future_horizons=(1,),
        )

        train, valid, test = chronological_split_indices(
            dataset,
            train_end=70,
            valid_end=160,
            purge=1,
        )

        self.assertEqual(
            [dataset[index].decision_date for index in train], [10, 20, 40]
        )
        self.assertEqual([dataset[index].decision_date for index in valid], [110])
        self.assertEqual([dataset[index].decision_date for index in test], [220])
        self.assertTrue(set(train).isdisjoint(valid))
        self.assertTrue(set(valid).isdisjoint(test))

    def test_chronological_split_accepts_string_boundaries_for_yyyymmdd_dates(
        self,
    ) -> None:
        factors = pd.DataFrame(
            {
                "date": [
                    20080101,
                    20201231,
                    20210101,
                    20221231,
                    20230101,
                    20250101,
                    20260101,
                ],
                "asset": ["A"] * 7,
                "value": list(range(7)),
            }
        )
        dataset = PanelFrameDataset(
            factors,
            context_length=1,
            future_horizons=(1,),
        )

        train, valid, test = chronological_split_indices(
            dataset,
            train_end="2020-12-31",
            valid_end="2022-12-31",
            purge=0,
        )

        self.assertEqual(
            [dataset[index].decision_date for index in train], [20080101, 20201231]
        )
        self.assertEqual(
            [dataset[index].decision_date for index in valid], [20210101, 20221231]
        )
        self.assertEqual(
            [dataset[index].decision_date for index in test], [20230101, 20250101]
        )


if __name__ == "__main__":
    unittest.main()
