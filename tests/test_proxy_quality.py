import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from factorpanel_data.proxy_config import ProxyFactorConfig
from factorpanel_data.proxy_materialize import materialize_year, sha256_file


class QualityProvider:
    def query(self, fields, names, start_time, end_time):
        year = int(str(start_time)[:4])
        dates = pd.to_datetime([f"{year}-01-02", f"{year}-01-03", f"{year}-01-04"])
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


class FormulaAuditProvider:
    def __init__(
        self,
        *,
        corrupt_factor: str | None = None,
        null_factor: str | None = None,
    ) -> None:
        self.corrupt_factor = corrupt_factor
        self.null_factor = null_factor

    @staticmethod
    def _raw_frame(year: int) -> pd.DataFrame:
        dates = pd.bdate_range(f"{year}-01-02", periods=35)
        index = pd.MultiIndex.from_product(
            [["SH600000", "SZ000001"], dates],
            names=["instrument", "datetime"],
        )
        step = np.tile(np.arange(len(dates), dtype="float64"), 2)
        asset_offset = np.repeat([0.0, 7.0], len(dates))
        close = 10.0 + asset_offset + step * 0.2
        open_ = close * (0.99 + (step % 3) * 0.002)
        high = np.maximum(open_, close) * 1.02
        low = np.minimum(open_, close) * 0.98
        volume = 1000.0 + asset_offset * 10 + step * 13
        vwap = (open_ + high + low + close) / 4
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "vwap": vwap,
                "volume": volume,
                "amount": vwap * volume,
            },
            index=index,
        )

    def query(self, fields, names, start_time, end_time):
        from factorpanel_data.proxy_quality import compute_proxy_factors_pandas

        year = int(str(start_time)[:4])
        raw = self._raw_frame(year)
        if names == ["open", "high", "low", "close", "vwap", "volume", "amount"]:
            return raw.loc[:, names]
        parts = []
        for _, asset_raw in raw.groupby(level="instrument", sort=False):
            flat = asset_raw.droplevel("instrument")
            calculated = compute_proxy_factors_pandas(flat)
            calculated.index = pd.MultiIndex.from_product(
                [[asset_raw.index[0][0]], calculated.index],
                names=["instrument", "datetime"],
            )
            parts.append(calculated)
        factors = pd.concat(parts).sort_index().loc[:, names]
        if self.corrupt_factor in factors:
            factors[self.corrupt_factor] += 0.01
        if self.null_factor in factors:
            factors.loc[factors.index[-1], self.null_factor] = np.nan
        return factors


def passing_causality_audit(config: ProxyFactorConfig) -> dict:
    from factorpanel_data.proxy_quality import FORMULA_AUDIT_FACTORS

    years = list(config.years[: min(3, len(config.years))])
    return {
        "passed": True,
        "fingerprint": config.fingerprint,
        "pandas_perturbation": {"passed": True, "factors_checked": 128},
        "qlib_vs_pandas": {
            "passed": True,
            "years_checked": years,
            "factors_checked": list(FORMULA_AUDIT_FACTORS),
            "comparisons": 100,
            "max_abs_error": 0.0,
            "registry_causal": True,
        },
    }


def quality_config(root: Path, end_year: int = 2008) -> ProxyFactorConfig:
    provider = root / "qlib"
    provider.mkdir(exist_ok=True)
    instruments = provider / "instruments"
    instruments.mkdir()
    (instruments / "all.txt").write_text(
        "SH600000\t2008-01-01\t2008-12-31\nSZ000001\t2008-01-01\t2008-12-31\n",
        encoding="utf-8",
    )
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
                "date": pd.to_datetime(["2008-01-02", "2008-01-02", "2008-01-03"]),
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

            manifest = finalize_dataset(
                config,
                causality_audit=passing_causality_audit(config),
            )

            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["years"], [2008])
            self.assertEqual(len(manifest["factors"]), 128)
            self.assertEqual(len(manifest["labels"]), 3)
            quality_path = config.output_root / "quality_report.json"
            manifest_path = config.output_root / "manifest.json"
            self.assertTrue(quality_path.is_file())
            self.assertTrue(manifest_path.is_file())
            on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                on_disk["quality_report_sha256"], manifest["quality_report_sha256"]
            )
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            self.assertEqual(len(quality["yearly_factors"]["2008"]), 128)
            self.assertTrue(quality["causality_audit"]["passed"])
            self.assertEqual(
                manifest["partitions"]["2008"]["factor_bytes"],
                (config.output_root / "factors/year=2008/part.parquet").stat().st_size,
            )
            self.assertEqual(
                manifest["partitions"]["2008"]["label_bytes"],
                (config.output_root / "labels/year=2008/part.parquet").stat().st_size,
            )

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

    def test_qlib_registry_matches_independent_pandas_for_three_years(self):
        from factorpanel_data.proxy_quality import verify_qlib_formula_audit

        with tempfile.TemporaryDirectory() as directory:
            config = quality_config(Path(directory), end_year=2010)

            result = verify_qlib_formula_audit(
                config,
                FormulaAuditProvider(),
                years=(2008, 2009, 2010),
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["years_checked"], [2008, 2009, 2010])
            self.assertEqual(len(result["factors_checked"]), 24)
            self.assertGreater(result["comparisons"], 0)
            self.assertTrue(result["registry_causal"])

    def test_qlib_registry_audit_detects_formula_divergence(self):
        from factorpanel_data.proxy_quality import verify_qlib_formula_audit

        with tempfile.TemporaryDirectory() as directory:
            config = quality_config(Path(directory), end_year=2010)

            result = verify_qlib_formula_audit(
                config,
                FormulaAuditProvider(corrupt_factor="pf_return_1"),
                years=(2008, 2009, 2010),
            )

            self.assertFalse(result["passed"])
            self.assertIn("pf_return_1", result["error"])

    def test_qlib_registry_audit_detects_one_sided_nan(self):
        from factorpanel_data.proxy_quality import verify_qlib_formula_audit

        with tempfile.TemporaryDirectory() as directory:
            config = quality_config(Path(directory), end_year=2010)

            result = verify_qlib_formula_audit(
                config,
                FormulaAuditProvider(null_factor="pf_return_1"),
                years=(2008, 2009, 2010),
            )

            self.assertFalse(result["passed"])
            self.assertIn("finite mask", result["error"])

    def test_finalize_requires_recorded_causality_audit(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            materialize_year(config, QualityProvider(), 2008)

            with self.assertRaisesRegex(ValueError, "causality audit"):
                finalize_dataset(config)

            self.assertFalse((config.output_root / "manifest.json").exists())

    def test_finalize_rejects_incomplete_formula_audit(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            materialize_year(config, QualityProvider(), 2008)
            audit = passing_causality_audit(config)
            audit["qlib_vs_pandas"]["factors_checked"] = audit["qlib_vs_pandas"][
                "factors_checked"
            ][:10]

            with self.assertRaisesRegex(ValueError, "Qlib/pandas evidence"):
                finalize_dataset(config, causality_audit=audit)

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

    def test_pandas_rolling_std_matches_qlib_sample_std(self):
        from factorpanel_data.proxy_quality import compute_proxy_factors_pandas

        close = pd.Series(np.linspace(10.0, 20.0, 25))
        volume = pd.Series(np.linspace(100.0, 300.0, 25))
        raw = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "vwap": close,
                "volume": volume,
                "amount": close * volume,
            }
        )

        actual = compute_proxy_factors_pandas(raw)
        expected_price = close.rolling(20, min_periods=20).std() / close
        expected_volume = (
            volume.rolling(20, min_periods=20).std()
            / volume.rolling(20, min_periods=20).mean()
        )

        np.testing.assert_allclose(actual["pf_std_20"], expected_price, equal_nan=True)
        np.testing.assert_allclose(
            actual["pf_vstd_20"], expected_volume, equal_nan=True
        )

    def test_finalize_rejects_checksum_consistent_factor_label_key_mismatch(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            result = materialize_year(config, QualityProvider(), 2008)
            labels = pd.read_parquet(result.label_path)
            labels.loc[0, "asset"] = "SH999999"
            labels.to_parquet(result.label_path, index=False, compression="zstd")
            state_path = config.output_root / "_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["years"]["2008"]["label_sha256"] = sha256_file(result.label_path)
            state_path.write_text(json.dumps(state), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "factor/label keys"):
                finalize_dataset(config)

    def test_finalize_rejects_incorrect_label_mask(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            result = materialize_year(config, QualityProvider(), 2008)
            labels = pd.read_parquet(result.label_path)
            labels.loc[0, "ret_1d_mask"] = False
            labels.to_parquet(result.label_path, index=False, compression="zstd")
            state_path = config.output_root / "_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["years"]["2008"]["label_sha256"] = sha256_file(result.label_path)
            state_path.write_text(json.dumps(state), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mask"):
                finalize_dataset(config)

    def test_finalize_rejects_non_boolean_label_mask_schema(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            result = materialize_year(config, QualityProvider(), 2008)
            labels = pd.read_parquet(result.label_path)
            labels["ret_1d_mask"] = labels["ret_1d_mask"].astype("int8")
            labels.to_parquet(result.label_path, index=False, compression="zstd")
            state_path = config.output_root / "_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["years"]["2008"]["label_sha256"] = sha256_file(result.label_path)
            state_path.write_text(json.dumps(state), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "schema"):
                finalize_dataset(config)

    def test_finalize_rejects_state_row_count_mismatch(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            materialize_year(config, QualityProvider(), 2008)
            state_path = config.output_root / "_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["years"]["2008"]["factor_rows"] += 1
            state_path.write_text(json.dumps(state), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "row count"):
                finalize_dataset(config)

    def test_finalize_rejects_rows_outside_listing_interval(self):
        from factorpanel_data.proxy_quality import finalize_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = quality_config(root)
            materialize_year(config, QualityProvider(), 2008)
            (config.provider_uri / "instruments/all.txt").write_text(
                "SH600000\t2008-01-03\t2008-12-31\nSZ000001\t2008-01-03\t2008-12-31\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "listing interval"):
                finalize_dataset(config)


if __name__ == "__main__":
    unittest.main()
