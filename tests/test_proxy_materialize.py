import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from factorpanel_data.proxy_config import ProxyFactorConfig


class MaterializeProvider:
    def __init__(self):
        self.calls = 0

    def query(self, fields, names, start_time, end_time):
        self.calls += 1
        year = int(str(start_time)[:4])
        dates = pd.to_datetime([f"{year}-01-02", f"{year}-01-03"])
        index = pd.MultiIndex.from_product(
            [["SH600000", "SZ000001"], dates],
            names=["instrument", "datetime"],
        )
        values = {
            name: np.arange(len(index), dtype=np.float64) + offset / 10
            for offset, name in enumerate(names)
        }
        return pd.DataFrame(values, index=index)


def make_config(root: Path) -> ProxyFactorConfig:
    provider = root / "qlib"
    provider.mkdir(exist_ok=True)
    return ProxyFactorConfig(
        provider_uri=provider,
        output_root=root / "proxy",
        start_year=2008,
        end_year=2025,
        factor_shard_size=32,
    )


class ProxyMaterializeTests(unittest.TestCase):
    def test_normalize_orders_keys_casts_values_and_replaces_infinity(self):
        from factorpanel_data.proxy_materialize import normalize_proxy_frame

        index = pd.MultiIndex.from_tuples(
            [
                ("SZ000001", pd.Timestamp("2008-01-03")),
                ("SH600000", pd.Timestamp("2008-01-02")),
            ],
            names=["instrument", "datetime"],
        )
        source = pd.DataFrame(
            {"factor_a": [np.inf, 2.0], "factor_b": [3.0, 4.0]},
            index=index,
        )

        result = normalize_proxy_frame(source, ("factor_a", "factor_b"))

        self.assertEqual(result.columns.tolist(), ["date", "asset", "factor_a", "factor_b"])
        self.assertEqual(result["asset"].tolist(), ["SH600000", "SZ000001"])
        self.assertEqual(str(result["factor_a"].dtype), "float32")
        self.assertTrue(pd.isna(result.loc[1, "factor_a"]))

    def test_materialize_year_publishes_both_partitions_and_state(self):
        from factorpanel_data.proxy_materialize import materialize_year, sha256_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)

            result = materialize_year(config, MaterializeProvider(), 2008)

            self.assertEqual(
                result.factor_path,
                config.output_root / "factors/year=2008/part.parquet",
            )
            self.assertEqual(
                result.label_path,
                config.output_root / "labels/year=2008/part.parquet",
            )
            self.assertTrue(result.factor_path.is_file())
            self.assertTrue(result.label_path.is_file())
            self.assertEqual(result.factor_rows, 4)
            self.assertEqual(result.factor_columns, 128)
            labels = pd.read_parquet(result.label_path)
            self.assertEqual(
                labels.columns.tolist(),
                [
                    "date",
                    "asset",
                    "ret_1d",
                    "ret_5d",
                    "ret_20d",
                    "ret_1d_mask",
                    "ret_5d_mask",
                    "ret_20d_mask",
                ],
            )
            self.assertTrue(
                all(str(labels[column].dtype) == "bool" for column in labels if column.endswith("_mask"))
            )
            self.assertFalse(list(config.output_root.rglob("*.tmp")))
            state = json.loads(
                (config.output_root / "_state.json").read_text(encoding="utf-8")
            )
            year_state = state["years"]["2008"]
            self.assertEqual(year_state["factor_sha256"], sha256_file(result.factor_path))
            self.assertEqual(year_state["label_sha256"], sha256_file(result.label_path))

    def test_writer_failure_does_not_replace_existing_partitions(self):
        from factorpanel_data.proxy_materialize import materialize_year, sha256_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            factor_path = config.output_root / "factors/year=2008/part.parquet"
            label_path = config.output_root / "labels/year=2008/part.parquet"
            factor_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)
            factor_path.write_bytes(b"existing-factor")
            label_path.write_bytes(b"existing-label")
            factor_before = sha256_file(factor_path)
            label_before = sha256_file(label_path)
            calls = 0

            def failing_writer(frame, destination, compression):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected writer failure")
                frame.to_parquet(destination, index=False, compression=compression)

            with self.assertRaisesRegex(RuntimeError, "injected writer failure"):
                materialize_year(
                    config,
                    MaterializeProvider(),
                    2008,
                    writer=failing_writer,
                    force=True,
                )

            self.assertEqual(sha256_file(factor_path), factor_before)
            self.assertEqual(sha256_file(label_path), label_before)
            self.assertFalse(list(config.output_root.rglob("*.tmp")))

    def test_resume_skips_only_when_fingerprint_and_checksums_match(self):
        from factorpanel_data.proxy_materialize import materialize_year

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            first_provider = MaterializeProvider()
            materialize_year(config, first_provider, 2008)
            second_provider = MaterializeProvider()

            result = materialize_year(config, second_provider, 2008, resume=True)

            self.assertTrue(result.skipped)
            self.assertEqual(second_provider.calls, 0)

    def test_resume_generates_a_year_not_present_in_valid_state(self):
        from factorpanel_data.proxy_materialize import materialize_year

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            materialize_year(config, MaterializeProvider(), 2008)
            provider = MaterializeProvider()

            result = materialize_year(config, provider, 2009, resume=True)

            self.assertFalse(result.skipped)
            self.assertEqual(result.year, 2009)
            self.assertEqual(provider.calls, 5)

    def test_resume_rejects_checksum_mismatch(self):
        from factorpanel_data.proxy_materialize import materialize_year

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            result = materialize_year(config, MaterializeProvider(), 2008)
            result.factor_path.write_bytes(b"corrupted")

            with self.assertRaisesRegex(ValueError, "checksum"):
                materialize_year(config, MaterializeProvider(), 2008, resume=True)


if __name__ == "__main__":
    unittest.main()
