import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd


class FakeProxyProvider:
    def __init__(self):
        self.calls = []

    def query(self, fields, names, start_time, end_time):
        self.calls.append(
            {
                "fields": tuple(fields),
                "names": tuple(names),
                "start_time": str(start_time),
                "end_time": str(end_time),
            }
        )
        dates = pd.to_datetime(["2007-12-31", "2008-01-02", "2008-01-03"])
        index = pd.MultiIndex.from_product(
            [["SH600000", "SZ000001"], dates],
            names=["instrument", "datetime"],
        )
        data = {
            name: np.arange(len(index), dtype=np.float64) + offset
            for offset, name in enumerate(names)
        }
        return pd.DataFrame(data, index=index)


class ProxyFactorConfigTests(unittest.TestCase):
    def test_config_api_is_exported_from_package(self):
        from factorpanel_data import ProxyFactorConfig, load_proxy_config

        self.assertTrue(callable(ProxyFactorConfig))
        self.assertTrue(callable(load_proxy_config))

    def test_default_config_covers_full_requested_scope(self):
        from factorpanel_data.proxy_config import ProxyFactorConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = root / "qlib"
            provider.mkdir()
            config = ProxyFactorConfig(
                provider_uri=provider,
                output_root=root / "proxy",
                start_year=2008,
                end_year=2025,
                universe="all",
                factor_shard_size=32,
            )

            self.assertEqual(config.years, tuple(range(2008, 2026)))
            self.assertEqual(config.factor_shard_size, 32)
            self.assertEqual(len(config.fingerprint), 64)

    def test_config_fingerprint_changes_with_generation_contract(self):
        from factorpanel_data.proxy_config import ProxyFactorConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = root / "qlib"
            provider.mkdir()
            first = ProxyFactorConfig(provider, root / "a", factor_shard_size=32)
            second = ProxyFactorConfig(provider, root / "a", factor_shard_size=16)

            self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_load_config_resolves_paths_from_project_root(self):
        from factorpanel_data.proxy_config import load_proxy_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "qlib").mkdir(parents=True)
            config_path = root / "configs" / "proxy.json"
            config_path.parent.mkdir()
            config_path.write_text(
                json.dumps(
                    {
                        "provider_uri": "data/qlib",
                        "output_root": "data/proxy",
                        "start_year": 2008,
                        "end_year": 2025,
                        "universe": "all",
                        "factor_shard_size": 32,
                    }
                ),
                encoding="utf-8",
            )

            config = load_proxy_config(config_path, project_root=root)

            self.assertEqual(config.provider_uri, (root / "data/qlib").resolve())
            self.assertEqual(config.output_root, (root / "data/proxy").resolve())

    def test_config_rejects_invalid_shard_size(self):
        from factorpanel_data.proxy_config import ProxyFactorConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = root / "qlib"
            provider.mkdir()
            with self.assertRaisesRegex(ValueError, "divide 128"):
                ProxyFactorConfig(provider, root / "proxy", factor_shard_size=30)


class QlibProxyQueryTests(unittest.TestCase):
    def test_real_provider_defaults_to_single_process(self):
        from factorpanel_data.qlib_proxy import QlibProxyProvider

        calls = []
        fake_qlib = types.ModuleType("qlib")
        fake_qlib.__path__ = []
        fake_qlib.init = lambda **kwargs: calls.append(kwargs)
        fake_config = types.ModuleType("qlib.config")
        fake_config.REG_CN = "cn"
        fake_data = types.ModuleType("qlib.data")
        fake_data.D = types.SimpleNamespace(
            features=object(),
            instruments=lambda universe: (universe,),
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "qlib": fake_qlib,
                "qlib.config": fake_config,
                "qlib.data": fake_data,
            },
        ):
            QlibProxyProvider(".")

        self.assertEqual(calls[0]["kernels"], 1)

    def test_query_year_joins_shards_and_trims_to_target_year(self):
        from factorpanel_data.qlib_proxy import query_factor_year

        provider = FakeProxyProvider()

        frame = query_factor_year(provider, year=2008, shard_size=32)

        self.assertEqual(frame.index.names, ["instrument", "datetime"])
        self.assertEqual(frame.shape[1], 128)
        self.assertEqual(
            frame.index.get_level_values("datetime").year.unique().tolist(),
            [2008],
        )
        self.assertEqual(len(provider.calls), 4)
        self.assertTrue(all(len(call["fields"]) == 32 for call in provider.calls))

    def test_query_label_year_is_separate_and_trims_target_year(self):
        from factorpanel_data.qlib_proxy import query_label_year

        provider = FakeProxyProvider()

        frame = query_label_year(provider, year=2008)

        self.assertEqual(frame.columns.tolist(), ["ret_1d", "ret_5d", "ret_20d"])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            frame.index.get_level_values("datetime").year.unique().tolist(),
            [2008],
        )

    def test_duplicate_provider_index_is_rejected(self):
        from factorpanel_data.qlib_proxy import query_factor_year

        class DuplicateProvider(FakeProxyProvider):
            def query(self, fields, names, start_time, end_time):
                frame = super().query(fields, names, start_time, end_time)
                return pd.concat([frame, frame.iloc[[0]]])

        with self.assertRaisesRegex(ValueError, "duplicate index"):
            query_factor_year(DuplicateProvider(), year=2008, shard_size=32)


if __name__ == "__main__":
    unittest.main()
