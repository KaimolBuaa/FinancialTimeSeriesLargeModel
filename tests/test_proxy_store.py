import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


class ProxyFactorStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        factor_root = cls.root / "factors/year=2020"
        factor_root.mkdir(parents=True)
        dates = pd.bdate_range("2020-01-01", periods=300)
        assets = [f"SH{600000 + index:06d}" for index in range(520)]
        index = pd.MultiIndex.from_product(
            [dates, assets],
            names=["date", "asset"],
        )
        rows = len(index)
        values = np.arange(rows, dtype="float32") / 10_000
        values[::997] = np.nan
        frame = index.to_frame(index=False)
        frame["pf_roc_20"] = values
        frame["pf_decoy"] = np.float32(123.0)
        frame.to_parquet(factor_root / "part.parquet", index=False)
        (cls.root / "manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "years": [2020],
                    "factors": [
                        {"name": "pf_roc_20"},
                        {"name": "pf_decoy"},
                    ],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_store_reads_one_factor_with_date_and_asset_filters(self):
        from factorpanel_data.proxy_store import ProxyFactorStore

        store = ProxyFactorStore(self.root)
        frame = store.read_factor(
            "pf_roc_20",
            start_date="2020-01-01",
            end_date="2020-03-31",
            assets=("SH600000", "SH600001"),
        )

        self.assertEqual(frame.columns.tolist(), ["date", "asset", "pf_roc_20"])
        self.assertEqual(set(frame["asset"]), {"SH600000", "SH600001"})
        self.assertGreaterEqual(frame["date"].min(), pd.Timestamp("2020-01-01"))
        self.assertLessEqual(frame["date"].max(), pd.Timestamp("2020-03-31"))

    def test_store_builds_bounded_model_panel(self):
        from factorpanel_data.proxy_store import ProxyFactorStore

        panel = ProxyFactorStore(self.root).read_panel(
            factor="pf_roc_20",
            end_date="2021-02-23",
            context_length=256,
            max_assets=512,
            seed=7,
        )

        self.assertEqual(panel.values.shape, (256, 512))
        self.assertEqual(panel.observed_mask.shape, (256, 512))
        self.assertEqual(panel.observed_mask.dtype, np.bool_)
        self.assertTrue(np.isfinite(panel.values).all())
        self.assertEqual(len(panel.dates), 256)
        self.assertEqual(len(panel.assets), 512)
        self.assertEqual(tuple(sorted(panel.assets)), panel.assets)

    def test_panel_asset_sampling_is_reproducible(self):
        from factorpanel_data.proxy_store import ProxyFactorStore

        store = ProxyFactorStore(self.root)
        first = store.read_panel(
            factor="pf_roc_20",
            end_date="2021-02-23",
            context_length=256,
            max_assets=32,
            seed=19,
        )
        second = store.read_panel(
            factor="pf_roc_20",
            end_date="2021-02-23",
            context_length=256,
            max_assets=32,
            seed=19,
        )

        self.assertEqual(first.assets, second.assets)
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(first.observed_mask, second.observed_mask)

    def test_store_rejects_factor_not_in_manifest(self):
        from factorpanel_data.proxy_store import ProxyFactorStore

        with self.assertRaisesRegex(ValueError, "unknown factor"):
            ProxyFactorStore(self.root).read_factor(
                "pf_missing",
                start_date="2020-01-01",
                end_date="2020-01-31",
            )

    def test_public_store_api_is_exported(self):
        from factorpanel_data import ProxyFactorStore

        self.assertTrue(callable(ProxyFactorStore))


if __name__ == "__main__":
    unittest.main()
