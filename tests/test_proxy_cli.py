import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from factorpanel_data.proxy_config import load_proxy_config
from factorpanel_data.proxy_materialize import materialize_year


ROOT = Path(__file__).resolve().parents[1]


class CliProvider:
    def query(self, fields, names, start_time, end_time):
        year = int(str(start_time)[:4])
        dates = pd.to_datetime([f"{year}-01-02", f"{year}-01-03"])
        index = pd.MultiIndex.from_product(
            [["SH600000", "SZ000001"], dates],
            names=["instrument", "datetime"],
        )
        return pd.DataFrame(
            {
                name: np.arange(len(index), dtype=np.float64) + offset / 100
                for offset, name in enumerate(names)
            },
            index=index,
        )


def write_config(root: Path, end_year: int = 2025) -> Path:
    provider = root / "qlib"
    provider.mkdir()
    output = root / "proxy"
    config_path = root / "configs" / "proxy.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "provider_uri": str(provider),
                "output_root": str(output),
                "start_year": 2008,
                "end_year": end_year,
                "universe": "all",
                "frequency": "day",
                "warmup_trading_days": 120,
                "factor_shard_size": 32,
                "compression": "zstd",
                "min_global_valid_ratio": 0.05,
                "max_near_constant_ratio": 0.99,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def run_cli(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "factorpanel_data.proxy_cli", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=check,
    )


class ProxyCliTests(unittest.TestCase):
    def test_status_reports_all_missing_years_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_config(Path(directory))

            result = run_cli("status", "--config", str(config), "--json")
            payload = json.loads(result.stdout)

            self.assertTrue(payload["ok"])
            self.assertFalse(payload["complete"])
            self.assertEqual(payload["missing_factor_years"], list(range(2008, 2026)))
            self.assertEqual(payload["missing_label_years"], list(range(2008, 2026)))

    def test_generate_rejects_mismatched_state_before_importing_qlib(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_config(root)
            output = root / "proxy"
            output.mkdir()
            (output / "_state.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "fingerprint": "wrong",
                        "years": {"2008": {}},
                    }
                ),
                encoding="utf-8",
            )

            result = run_cli(
                "generate",
                "--config",
                str(config),
                "--resume",
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            error = json.loads(result.stderr.strip().splitlines()[-1])
            self.assertFalse(error["ok"])
            self.assertIn("fingerprint", error["error"])
            self.assertNotIn("pyqlib", error["error"])

    def test_help_lists_operational_commands(self):
        result = run_cli("--help")

        self.assertIn("generate", result.stdout)
        self.assertIn("verify", result.stdout)
        self.assertIn("status", result.stdout)
        self.assertIn("sample-query", result.stdout)

    def test_verify_year_reports_schema_keys_nonfinite_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_config(root, end_year=2008)
            config = load_proxy_config(config_path, project_root=root)
            materialize_year(config, CliProvider(), 2008)

            result = run_cli(
                "verify",
                "--config",
                str(config_path),
                "--year",
                "2008",
            )
            payload = json.loads(result.stdout)

            self.assertTrue(payload["ok"])
            report = payload["years"]["2008"]
            self.assertEqual(report["factor_columns"], 128)
            self.assertEqual(report["duplicate_keys"], 0)
            self.assertEqual(report["nonfinite_values"], 0)
            self.assertTrue(report["checksums_valid"])

    def test_pyproject_registers_proxy_console_script(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('factorpanel-proxy = "factorpanel_data.proxy_cli:main"', pyproject)

    def test_full_force_set_resets_mismatched_state_with_backup(self):
        from factorpanel_data.proxy_cli import prepare_for_forced_migration

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_config(root, end_year=2009)
            config = load_proxy_config(config_path, project_root=root)
            config.output_root.mkdir()
            state_path = config.output_root / "_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "fingerprint": "old-fingerprint",
                        "years": {"2008": {}, "2009": {}},
                    }
                ),
                encoding="utf-8",
            )

            backup = prepare_for_forced_migration(config, {2008, 2009})

            self.assertIsNotNone(backup)
            self.assertTrue(backup.is_file())
            old_state = json.loads(backup.read_text(encoding="utf-8"))
            new_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(old_state["fingerprint"], "old-fingerprint")
            self.assertEqual(new_state["fingerprint"], config.fingerprint)
            self.assertEqual(new_state["years"], {})


if __name__ == "__main__":
    unittest.main()
