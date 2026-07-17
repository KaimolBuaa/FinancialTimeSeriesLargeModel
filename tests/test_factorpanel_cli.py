from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorpanel_fm.cli import _module  # noqa: E402
from factorpanel_fm.model import ModelConfig  # noqa: E402


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "factorpanel_fm", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


class FactorPanelCliTests(unittest.TestCase):
    def test_smoke_stage_a_and_b_emit_one_json_line_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for stage in ("a", "b"):
                with self.subTest(stage=stage):
                    checkpoint = Path(directory) / f"stage-{stage}.pt"
                    result = run_cli(
                        "smoke",
                        "--stage",
                        stage,
                        "--steps",
                        "1",
                        "--device",
                        "cpu",
                        "--seed",
                        "9",
                        "--checkpoint",
                        str(checkpoint),
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(len(result.stdout.strip().splitlines()), 1)
                    summary = json.loads(result.stdout)
                    self.assertEqual(summary["stage"], stage)
                    self.assertEqual(summary["model"], "tiny")
                    self.assertEqual(summary["step"], 1)
                    self.assertEqual(summary["device"], "cpu")
                    self.assertTrue(checkpoint.is_file())
                    if stage == "b":
                        payload = torch.load(checkpoint, weights_only=True)
                        groups = payload["optimizer_state"]["param_groups"]
                        self.assertEqual(len(groups), 4)
                        self.assertEqual(
                            payload["metadata"]["stage_config"]["horizons"],
                            (1, 5, 20),
                        )
                        self.assertFalse(
                            any(
                                key.startswith("module.")
                                for key in payload["model_state"]
                            )
                        )
                        self.assertTrue(
                            all(
                                group["lr"] == 0.0
                                for group in groups
                                if group["stage_b_lower"]
                            )
                        )

    def test_pilot_reads_parquet_and_runs_one_step(self) -> None:
        factor_rows = []
        label_rows = []
        for date in range(1, 38):
            for asset in range(4):
                factor_rows.append(
                    {"date": date, "asset": asset, "alpha": date + asset / 10}
                )
                label_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "r1": asset - 1.5,
                        "r5": date / 10 + asset,
                        "r20": date / 20 - asset,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            factors = Path(directory) / "factors.parquet"
            labels = Path(directory) / "labels.parquet"
            checkpoint = Path(directory) / "pilot.pt"
            pd.DataFrame(factor_rows).to_parquet(factors)
            pd.DataFrame(label_rows).to_parquet(labels)

            result = run_cli(
                "pilot",
                "--factors",
                str(factors),
                "--labels",
                str(labels),
                "--factor-columns",
                "alpha",
                "--return-columns",
                "r1,r5,r20",
                "--return-horizons",
                "1,5,20",
                "--stage",
                "b",
                "--model",
                "tiny",
                "--context-length",
                "16",
                "--steps",
                "1",
                "--grad-accum",
                "1",
                "--device",
                "cpu",
                "--checkpoint",
                str(checkpoint),
            )
            checkpoint_created = checkpoint.is_file()

            mismatch = run_cli(
                "pilot",
                "--factors",
                str(factors),
                "--labels",
                str(labels),
                "--factor-columns",
                "alpha",
                "--return-columns",
                "r1,r5,r20",
                "--return-horizons",
                "1,5",
                "--stage",
                "b",
                "--model",
                "tiny",
                "--steps",
                "1",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["step"], 1)
        self.assertEqual(summary["model"], "tiny")
        self.assertTrue(checkpoint_created)
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("same length", mismatch.stderr)

    def test_stage_b_module_uses_explicit_return_horizons(self) -> None:
        module = _module("b", ModelConfig.tiny(), (2, 7, 30))

        self.assertEqual(module.config.horizons, (2, 7, 30))

    def test_bad_pilot_input_is_nonzero_and_clear(self) -> None:
        result = run_cli(
            "pilot",
            "--factors",
            "/does/not/exist.parquet",
            "--factor-columns",
            "alpha",
            "--stage",
            "a",
            "--steps",
            "1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exist", result.stderr.lower())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_mps_smoke_runs_one_step(self) -> None:
        result = run_cli("smoke", "--stage", "a", "--steps", "1", "--device", "mps")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["device"], "mps")


class FactorPanelSmallConfigTests(unittest.TestCase):
    def test_exact_model_objectives_and_server_pilot_values(self) -> None:
        config = json.loads((ROOT / "configs" / "factorpanel_small.json").read_text())
        self.assertEqual(
            config["model"],
            {
                "context_length": 256,
                "input_channels": 3,
                "patch_size": 16,
                "patch_stride": 8,
                "d_model": 384,
                "temporal_layers": 8,
                "num_heads": 8,
                "ffn_dim": 1536,
                "num_latents": 32,
                "set_layers": 2,
                "output_dim": 128,
                "dropout": 0.1,
                "use_set_mixer": True,
                "parameter_count": 21390848,
            },
        )
        self.assertEqual(config["stage_a"]["weights"], [1.0, 0.5, 0.1])
        self.assertEqual(config["stage_b"]["weights"], [1.0, 0.5, 0.2])
        pilot = config["server_pilot"]
        self.assertEqual(pilot["micro_batch_size"], 1)
        self.assertEqual(pilot["auto_profile_max_micro_batch_size"], 8)
        self.assertEqual(pilot["gradient_accumulation"], 4)
        self.assertTrue(pilot["bf16"])
        self.assertEqual(pilot["pilot_steps"], 20000)
        self.assertEqual(pilot["max_steps"], 200000)

    def test_environment_records_reproducible_runtime_without_qlib(self) -> None:
        import yaml

        environment = yaml.safe_load((ROOT / "environment.yml").read_text())
        self.assertEqual(environment["name"], "factorpanel-fm")
        dependencies = environment["dependencies"]
        self.assertIn("python=3.11", dependencies)
        pip = next(item["pip"] for item in dependencies if isinstance(item, dict))
        self.assertIn("torch==2.9.1", pip)
        self.assertIn("pandas==2.3.3", pip)
        self.assertIn("pyarrow==22.0.0", pip)
        self.assertIn("-e .", pip)
        self.assertFalse(any("qlib" in dependency.lower() for dependency in pip))

    def test_pyproject_declares_src_packages_and_core_dependencies(self) -> None:
        import tomllib

        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertGreaterEqual(project["project"]["requires-python"], ">=3.11")
        dependencies = project["project"]["dependencies"]
        self.assertTrue(any(value.startswith("torch") for value in dependencies))
        self.assertTrue(any(value.startswith("pandas") for value in dependencies))
        self.assertTrue(any(value.startswith("pyarrow") for value in dependencies))
        setuptools = project["tool"]["setuptools"]
        self.assertEqual(
            set(setuptools["packages"]),
            {"factorpanel_fm", "factorpanel_data"},
        )
        self.assertEqual(
            setuptools["package-dir"],
            {
                "factorpanel_fm": "src/factorpanel_fm",
                "factorpanel_data": "src/factorpanel_data",
            },
        )


if __name__ == "__main__":
    unittest.main()
