import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from factorpanel_data.proxy_config import ProxyFactorConfig
from factorpanel_data.proxy_materialize import materialize_year


class QualityProvider:
    def query(self, fields, names, start_time, end_time):
        year = int(str(start_time)[:4])
        dates = pd.to_datetime(
            [f"{year}-01-02", f"{year}-01-03", f"{year}-01-04"]
        )
        index = pd.MultiIndex.from_product(
            [["SH600000", "SZ000001"], dates],
            names=["instrument", "datetime"],
        )
        base = np.array(
            [
                timestamp.toordinal() + sum(ord(character) for character in asset)
                for asset, timestamp in index
            ],
            dtype=np.float64,
        )
        data = {
            name: base + sum(ord(character) for character in name) / 10_000
            for name in names
        }
        return pd.DataFrame(data, index=index)


class BoundaryProvider:
    def query(self, fields, names, start_time, end_time):
        end_year = int(str(end_time)[:4])
        dates = pd.to_datetime(
            [
                f"{end_year - 1}-12-03",
                f"{end_year}-01-02",
                f"{end_year}-01-03",
                f"{end_year}-01-04",
            ]
        )
        index = pd.MultiIndex.from_product(
            [["SH600000", "SZ000001"], dates],
            names=["instrument", "datetime"],
        )
        base = np.array(
            [
                timestamp.toordinal() + sum(ord(character) for character in asset)
                for asset, timestamp in index
            ],
            dtype=np.float64,
        )
        data = {
            name: base + sum(ord(character) for character in name) / 10_000
            for name in names
        }
        return pd.DataFrame(data, index=index)


def quality_config(root: Path, end_year: int = 2008) -> ProxyFactorConfig:
    provider = root / "qlib"
    provider.mkdir(exist_ok=True)
    return ProxyFactorConfig(
        provider_uri=provider,
        output_root=root / "proxy",
        start_year=2008,
        end_year=end_year,
        factor_shard_size=32,
        min_global_valid_ratio=0.05,
        max_near_constant_ratio=0.99,
    )


class ProxyQualityTests(unittest.TestCase):
    def test_partition_report_detects_nonfinite_duplicate_and_constant_data(self):
        from factorpanel_data.proxy_quality import inspect_factor_partition

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2008-01-02", "2008-01-02", "2008-01-03"]
                ),
                "asset": ["SH600000", "SH600000", "SZ000001"],
                "pf_constant": np.array([1.0, 1.0, 1.0], dtype="float32"),
                "pf_nonfinite": np.array([0.0, np.inf, 2.0], dtype="float32"),
            }
        )

        report = inspect_factor_partition(
            frame,
            expected_factors=("pf_constant", "pf_nonfinite"),
        )

        self.assertEqual(report.duplicate_keys, 1)
        self.assertEqual(report.nonfinite_values, 1)
        self.assertEqual(report.factors["pf_constant"].near_constant_ratio, 1.0)
        self.assertEqual(report.factors["pf_nonfinite"].valid_count, 2)

    def test_finalize_rejects_missing_years_without_publishing_manifest(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root, end_year=2025)

            with self.assertRaisesRegex(ValueError, "missing years"):
                finalize_dataset(config, completed_years=range(2008, 2025))

            self.assertFalse((config.output_root / "manifest.json").exists())

    def test_finalize_publishes_quality_report_then_manifest(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            materialize_year(config, QualityProvider(), 2008)

            manifest = finalize_dataset(config)

            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["years"], [2008])
            self.assertEqual(len(manifest["factors"]), 128)
            self.assertEqual(len(manifest["labels"]), 3)
            quality_path = config.output_root / "quality_report.json"
            manifest_path = config.output_root / "manifest.json"
            self.assertTrue(quality_path.is_file())
            self.assertTrue(manifest_path.is_file())
            on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["quality_report_sha256"], manifest["quality_report_sha256"])

    def test_finalize_rejects_checksum_corruption(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            result = materialize_year(config, QualityProvider(), 2008)
            result.factor_path.write_bytes(b"corrupt")

            with self.assertRaisesRegex(ValueError, "checksum"):
                finalize_dataset(config)

            self.assertFalse((config.output_root / "manifest.json").exists())

    def test_future_raw_data_perturbation_does_not_change_past_features(self):
        from factorpanel_data.proxy_quality import verify_pandas_causality

        result = verify_pandas_causality(seed=7, periods=260, cutoff=200)

        self.assertTrue(result["passed"])
        self.assertEqual(result["factors_checked"], 128)

    def test_year_boundary_matches_one_cross_year_query(self):
        from factorpanel_data.proxy_quality import verify_year_boundary

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            materialize_year(config, QualityProvider(), 2008)

            result = verify_year_boundary(config, BoundaryProvider(), 2008)

            self.assertTrue(result["passed"])
            self.assertEqual(result["factors_checked"], 14)
            self.assertEqual(result["dates_checked"], 3)

    def test_pandas_rsquared_two_uses_three_observation_minimum(self):
        from factorpanel_data.proxy_quality import compute_proxy_factors_pandas

        close = pd.Series([10.0, 11.0, 13.0, 12.0, 16.0, 15.0])
        raw = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "vwap": close * 1.001,
                "volume": [100.0, 120.0, 90.0, 140.0, 110.0, 160.0],
                "amount": close * 1.001 * 100.0,
            }
        )
        time = pd.Series(np.arange(len(close), dtype="float64"))
        expected = close.rolling(3, min_periods=3).corr(time).pow(2)

        actual = compute_proxy_factors_pandas(raw)["pf_rsqr_2"]

        np.testing.assert_allclose(
            actual.to_numpy(),
            expected.to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
