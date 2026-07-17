from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile
from urllib.error import HTTPError
import importlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PIPELINE = SRC / "factorpanel_data" / "pipeline.py"
ARCHIVE = SRC / "factorpanel_data" / "binance_um.py"
MATERIALIZE = SRC / "factorpanel_data" / "materialize.py"
CONFIG = SRC / "factorpanel_data" / "config.py"
BATCH = SRC / "factorpanel_data" / "batch.py"
CLI = SRC / "factorpanel_data" / "cli.py"


class CryptoPipelineTests(unittest.TestCase):
    def test_pipeline_module_exists(self):
        self.assertTrue(PIPELINE.is_file())

    def test_archive_module_exists(self):
        self.assertTrue(ARCHIVE.is_file())

    def test_materialize_module_exists(self):
        self.assertTrue(MATERIALIZE.is_file())

    def test_config_module_exists(self):
        self.assertTrue(CONFIG.is_file())

    def test_batch_module_exists(self):
        self.assertTrue(BATCH.is_file())

    def test_cli_reports_manifest_progress(self):
        sys.path.insert(0, str(SRC))
        module = importlib.import_module("factorpanel_data.cli")
        self.assertTrue(hasattr(module, "manifest_summary"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifests" / "raw_downloads.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(
                "{\"symbol\": \"BTCUSDT\", \"month\": \"2020-01\", \"status\": \"downloaded\"}\n"
                "{\"symbol\": \"ETHUSDT\", \"month\": \"2020-01\", \"status\": \"missing\"}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                module.manifest_summary(root),
                {"downloaded": 1, "missing": 1, "failed": 0},
            )

    @unittest.skipUnless(ARCHIVE.is_file(), "archive module has not been implemented")
    def test_archive_locations_and_storage_gate(self):
        sys.path.insert(0, str(SRC))
        from factorpanel_data.binance_um import (  # pylint: disable=import-outside-toplevel
            StorageLimitError,
            archive_path,
            archive_url,
            ensure_storage_budget,
        )

        self.assertEqual(
            archive_url("BTCUSDT", "2020-01"),
            "https://data.binance.vision/data/futures/um/monthly/klines/"
            "BTCUSDT/1m/BTCUSDT-1m-2020-01.zip",
        )
        self.assertEqual(
            archive_path(Path("data"), "BTCUSDT", "2020-01"),
            Path("data/raw_1m/BTCUSDT/2020/BTCUSDT-1m-2020-01.zip"),
        )
        ensure_storage_budget(
            current_bytes=50 * 1024**3,
            incoming_bytes=5 * 1024**3,
            target_bytes=60 * 1024**3,
            hard_bytes=80 * 1024**3,
        )
        with self.assertRaises(StorageLimitError):
            ensure_storage_budget(
                current_bytes=59 * 1024**3,
                incoming_bytes=3 * 1024**3,
                target_bytes=60 * 1024**3,
                hard_bytes=80 * 1024**3,
            )

    @unittest.skipUnless(MATERIALIZE.is_file(), "materialize module has not been implemented")
    def test_materialize_archive_writes_each_layer(self):
        sys.path.insert(0, str(SRC))
        from factorpanel_data.materialize import (  # pylint: disable=import-outside-toplevel
            materialize_archive,
        )

        minutes = pd.date_range("2020-01-01", periods=3_000, freq="min", tz="UTC")
        raw = pd.DataFrame(
            {
                0: (minutes.view("int64") // 1_000_000).astype("int64"),
                1: 100.0,
                2: 101.0,
                3: 99.0,
                4: 100.5,
                5: 2.0,
                6: (minutes.view("int64") // 1_000_000 + 59_999).astype("int64"),
                7: 200.0,
                8: 3,
                9: 1.0,
                10: 100.0,
                11: 0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "BTCUSDT-1m-2020-01.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("BTCUSDT-1m-2020-01.csv", raw.to_csv(index=False, header=False))

            outputs = materialize_archive(
                archive=archive,
                data_root=root / "data",
                symbol="BTCUSDT",
                source_month="2020-01",
                history_bars=3,
                liquidity_bars=2,
                liquidity_threshold=0.0,
            )
            self.assertEqual(set(outputs), {"canonical", "train_30m", "factors", "targets_masks"})
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            factors = pd.read_parquet(outputs["factors"])
            factor_columns = [column for column in factors if column.startswith("factor_")]
            self.assertEqual(len(factor_columns), 192)
            labels = pd.read_parquet(outputs["targets_masks"])
            self.assertTrue(labels.loc[0, "ret_1d_mask"])
            self.assertFalse(labels.loc[99, "ret_1d_mask"])

    @unittest.skipUnless(MATERIALIZE.is_file(), "materialize module has not been implemented")
    def test_materialize_archives_keeps_labels_across_month_boundary(self):
        sys.path.insert(0, str(SRC))
        module = importlib.import_module("factorpanel_data.materialize")
        self.assertTrue(hasattr(module, "materialize_archives"))
        materialize_archives = module.materialize_archives

        minutes = pd.date_range("2020-01-31", periods=8_000, freq="min", tz="UTC")
        close = 100 * np.exp(np.arange(8_000) / 100_000)
        raw = pd.DataFrame(
            {
                0: (minutes.view("int64") // 1_000_000).astype("int64"),
                1: close,
                2: close + 1,
                3: close - 1,
                4: close,
                5: 2.0,
                6: (minutes.view("int64") // 1_000_000 + 59_999).astype("int64"),
                7: 200.0,
                8: 3,
                9: 1.0,
                10: 100.0,
                11: 0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = []
            for month, rows in (("2020-01", raw.iloc[:4_000]), ("2020-02", raw.iloc[4_000:])):
                archive = root / f"BTCUSDT-1m-{month}.zip"
                with ZipFile(archive, "w") as zipped:
                    zipped.writestr(f"BTCUSDT-1m-{month}.csv", rows.to_csv(index=False, header=False))
                archives.append((archive, month))

            outputs = materialize_archives(
                archives=archives,
                data_root=root / "data",
                symbol="BTCUSDT",
                history_bars=3,
                liquidity_bars=2,
                liquidity_threshold=0.0,
            )
            labels = pd.read_parquet(outputs["targets_masks"])
            self.assertTrue(labels.loc[2, "ret_1d_mask"])
            self.assertTrue(labels.loc[2, "ret_5d_mask"])

    @unittest.skipUnless(MATERIALIZE.is_file(), "materialize module has not been implemented")
    def test_materialize_panel_computes_cross_sectional_factors(self):
        sys.path.insert(0, str(SRC))
        module = importlib.import_module("factorpanel_data.materialize")
        self.assertTrue(hasattr(module, "materialize_panel"))
        materialize_panel = module.materialize_panel

        minutes = pd.date_range("2020-01-01", periods=3_000, freq="min", tz="UTC")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives_by_symbol = {}
            for symbol, growth in (("BTCUSDT", 100_000), ("ETHUSDT", 200_000)):
                close = 100 * np.exp(np.arange(3_000) / growth)
                raw = pd.DataFrame(
                    {
                        0: (minutes.view("int64") // 1_000_000).astype("int64"),
                        1: close,
                        2: close + 1,
                        3: close - 1,
                        4: close,
                        5: 2.0,
                        6: (minutes.view("int64") // 1_000_000 + 59_999).astype("int64"),
                        7: 200.0,
                        8: 3,
                        9: 1.0,
                        10: 100.0,
                        11: 0,
                    }
                )
                archive = root / f"{symbol}-1m-2020-01.zip"
                with ZipFile(archive, "w") as zipped:
                    zipped.writestr(f"{symbol}-1m-2020-01.csv", raw.to_csv(index=False, header=False))
                archives_by_symbol[symbol] = [(archive, "2020-01")]

            outputs = materialize_panel(
                archives_by_symbol=archives_by_symbol,
                data_root=root / "data",
                history_bars=3,
                liquidity_bars=2,
                liquidity_threshold=0.0,
            )
            factors = pd.read_parquet(outputs["factors"])
            ranks = factors.loc[factors["timestamp"] == factors["timestamp"].iloc[-1], "factor_cs_rank_return_1"]
            self.assertEqual(sorted(ranks.tolist()), [0.5, 1.0])

    @unittest.skipUnless(CONFIG.is_file(), "config module has not been implemented")
    def test_config_months_are_inclusive_and_stable(self):
        sys.path.insert(0, str(SRC))
        from factorpanel_data.config import (  # pylint: disable=import-outside-toplevel
            PipelineConfig,
            iter_months,
        )

        config = PipelineConfig(
            data_root=Path("resources/data/binance_um"),
            symbols=("BTCUSDT", "ETHUSDT"),
            start_month="2020-01",
            end_month="2020-03",
            target_bytes=60 * 1024**3,
            hard_bytes=80 * 1024**3,
        )
        self.assertEqual(list(iter_months(config)), ["2020-01", "2020-02", "2020-03"])

    @unittest.skipUnless(BATCH.is_file(), "batch module has not been implemented")
    def test_batch_records_downloaded_and_missing_items(self):
        sys.path.insert(0, str(SRC))
        from factorpanel_data.batch import run_downloads  # pylint: disable=import-outside-toplevel
        from factorpanel_data.config import PipelineConfig  # pylint: disable=import-outside-toplevel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PipelineConfig(
                data_root=root / "data",
                symbols=("BTCUSDT", "NEWUSDT"),
                start_month="2020-01",
                end_month="2020-01",
                target_bytes=60 * 1024**3,
                hard_bytes=80 * 1024**3,
            )

            def fake_downloader(**kwargs):
                if kwargs["symbol"] == "NEWUSDT":
                    raise HTTPError("https://example.test", 404, "missing", None, None)
                return root / "BTCUSDT-1m-2020-01.zip"

            records = run_downloads(config, downloader=fake_downloader)
            self.assertEqual([record["status"] for record in records], ["downloaded", "missing"])
            repeated = run_downloads(config, downloader=fake_downloader)
            self.assertEqual(repeated, [])

    @unittest.skipUnless(PIPELINE.is_file(), "pipeline module has not been implemented")
    def test_canonicalization_resampling_and_labels_are_causal(self):
        sys.path.insert(0, str(SRC))
        from factorpanel_data.pipeline import (  # pylint: disable=import-outside-toplevel
            add_labels_and_masks,
            canonicalize_klines,
            resample_30m,
        )

        minutes = pd.date_range("2020-01-01", periods=60, freq="min", tz="UTC")
        raw = pd.DataFrame(
            {
                "open_time": (minutes.view("int64") // 1_000_000).astype("int64"),
                "open": np.arange(60, dtype=float) + 100,
                "high": np.arange(60, dtype=float) + 101,
                "low": np.arange(60, dtype=float) + 99,
                "close": np.arange(60, dtype=float) + 100.5,
                "volume": np.ones(60),
                "close_time": (minutes.view("int64") // 1_000_000 + 59_999).astype("int64"),
                "quote_volume": np.full(60, 10.0),
                "trade_count": np.ones(60, dtype=int),
                "taker_buy_base_volume": np.full(60, 0.4),
                "taker_buy_quote_volume": np.full(60, 4.0),
            }
        ).drop(index=45)

        canonical = canonicalize_klines(raw, symbol="BTCUSDT", source_month="2020-01")
        self.assertEqual(str(canonical["open_time_utc"].dt.tz), "UTC")

        bars = resample_30m(canonical)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars.loc[0, "minute_count"], 30)
        self.assertTrue(bars.loc[0, "bar_complete"])
        self.assertEqual(bars.loc[0, "base_volume"], 30.0)
        self.assertEqual(bars.loc[1, "minute_count"], 29)
        self.assertFalse(bars.loc[1, "bar_complete"])

        timestamps = pd.date_range("2020-01-01", periods=244, freq="30min", tz="UTC")
        complete = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": "BTCUSDT",
                "close": np.exp(np.arange(244) / 1_000),
                "quote_volume": 100.0,
                "bar_complete": True,
                "is_observed": True,
            }
        )
        labelled = add_labels_and_masks(
            complete,
            history_bars=3,
            liquidity_bars=2,
            liquidity_threshold=0.0,
        )
        self.assertAlmostEqual(labelled.loc[0, "ret_1d"], 48 / 1_000)
        self.assertAlmostEqual(labelled.loc[0, "ret_5d"], 240 / 1_000)
        self.assertTrue(pd.isna(labelled.loc[196, "ret_1d"]))
        self.assertTrue(pd.isna(labelled.loc[4, "ret_5d"]))
        self.assertFalse(labelled.loc[1, "liquidity_mask"])
        self.assertTrue(labelled.loc[2, "liquidity_mask"])
        self.assertTrue(labelled.loc[3, "trainable_mask"])


if __name__ == "__main__":
    unittest.main()
