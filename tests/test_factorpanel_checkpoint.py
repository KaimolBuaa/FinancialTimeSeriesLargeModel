from pathlib import Path
import copy
from collections import UserDict
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
    def assert_nested_equal(self, actual: object, expected: object) -> None:
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected)
        elif isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected))
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_nested_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)

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

    def test_load_uses_weights_only_and_rejects_unsupported_metadata(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, step=2, metadata={"tags": ["a", "b"]})
            with mock.patch(
                "factorpanel_fm.checkpoint.torch.load",
                wraps=torch.load,
            ) as loader:
                load_checkpoint(path, model, optimizer)
            self.assertTrue(loader.call_args.kwargs["weights_only"])

            with self.assertRaisesRegex((TypeError, ValueError), "metadata"):
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    step=2,
                    metadata={"unsupported": object()},
                )
            with self.assertRaisesRegex((TypeError, ValueError), "metadata"):
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    step=2,
                    metadata={"custom": UserDict({"value": 1})},
                )

    def test_incompatible_model_shape_or_dtype_does_not_mutate_objects(self) -> None:
        source = nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            shape_path = Path(directory) / "shape.pt"
            save_checkpoint(shape_path, source, source_optimizer, step=1)
            target = nn.Linear(3, 4)
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.7)
            expected_model = copy.deepcopy(target.state_dict())
            expected_optimizer = copy.deepcopy(target_optimizer.state_dict())

            with self.assertRaisesRegex(ValueError, "shape"):
                load_checkpoint(shape_path, target, target_optimizer)

            self.assert_nested_equal(target.state_dict(), expected_model)
            self.assert_nested_equal(target_optimizer.state_dict(), expected_optimizer)

            dtype_path = Path(directory) / "dtype.pt"
            double_source = nn.Linear(3, 2).double()
            double_optimizer = torch.optim.AdamW(double_source.parameters())
            save_checkpoint(dtype_path, double_source, double_optimizer, step=1)
            float_target = nn.Linear(3, 2)
            expected_float = copy.deepcopy(float_target.state_dict())
            with self.assertRaisesRegex(ValueError, "dtype"):
                load_checkpoint(dtype_path, float_target)
            self.assert_nested_equal(float_target.state_dict(), expected_float)

    def test_incompatible_optimizer_groups_do_not_mutate_model_or_optimizer(self) -> None:
        source = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.pt"
            save_checkpoint(path, source, source_optimizer, step=1)

            target = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
            parameters = list(target.parameters())
            target_optimizer = torch.optim.AdamW(
                [{"params": parameters[:2]}, {"params": parameters[2:]}],
                lr=0.7,
            )
            expected_model = copy.deepcopy(target.state_dict())
            expected_optimizer = copy.deepcopy(target_optimizer.state_dict())

            with self.assertRaisesRegex(ValueError, "optimizer param groups"):
                load_checkpoint(path, target, target_optimizer)

            self.assert_nested_equal(target.state_dict(), expected_model)
            self.assert_nested_equal(target_optimizer.state_dict(), expected_optimizer)


if __name__ == "__main__":
    unittest.main()
