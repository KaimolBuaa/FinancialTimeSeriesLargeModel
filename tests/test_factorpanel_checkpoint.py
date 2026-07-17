from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorpanel_fm.checkpoint import CheckpointInfo, load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_restores_model_optimizer_step_and_metadata(self) -> None:
        torch.manual_seed(21)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        expected_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "stage_b.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                step=37,
                metadata={"stage": "B", "dataset": "test"},
            )
            self.assertTrue(path.is_file())
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            optimizer.param_groups[0]["lr"] = 0.7

            info = load_checkpoint(path, model, optimizer)

        self.assertIsInstance(info, CheckpointInfo)
        self.assertEqual(info.step, 37)
        self.assertEqual(info.metadata, {"stage": "B", "dataset": "test"})
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_state[name])

    def test_failed_atomic_replace_removes_temporary_file(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new" / "checkpoint.pt"
            with mock.patch(
                "factorpanel_fm.checkpoint.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    save_checkpoint(path, model, optimizer, step=1)

            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob(".checkpoint.pt.*.tmp")), [])

    def test_load_supports_model_only_and_strict_flag(self) -> None:
        source = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(source.parameters(), lr=0.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, source, optimizer, step=4)
            target = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

            info = load_checkpoint(path, target, strict=False)

        self.assertEqual(info.step, 4)
        self.assertEqual(info.metadata, {})


if __name__ == "__main__":
    unittest.main()
