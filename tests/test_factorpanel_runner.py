from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorpanel_fm import (  # noqa: E402
    FactorPanelBatch,
    FactorPanelEncoder,
    FactorPanelSample,
    ModelConfig,
    RuntimeConfig,
    StageAModule,
    StageBConfig,
    StageBModule,
    auto_micro_batch_probe,
    resolve_device,
    run_training,
    warmup_cosine_multiplier,
)


def make_sample() -> FactorPanelSample:
    config = ModelConfig.tiny()
    values = torch.randn(1, config.context_length, 4)
    return FactorPanelSample(
        factor_id="alpha",
        batch=FactorPanelBatch(
            values=values,
            observed_mask=torch.ones_like(values, dtype=torch.bool),
            asset_ids=torch.arange(4).unsqueeze(0),
            dates=torch.arange(config.context_length).unsqueeze(0),
        ),
        future_factor_targets=torch.randn(1, 4, 2),
        future_factor_mask=torch.ones(1, 4, 2, dtype=torch.bool),
        return_targets=torch.randn(1, 4, 3),
        return_mask=torch.ones(1, 4, 3, dtype=torch.bool),
        decision_date=config.context_length - 1,
    )


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self) -> None:
        config = RuntimeConfig(stage="a")
        self.assertEqual(config.model, "tiny")
        self.assertEqual(config.micro_batch_size, 1)
        self.assertEqual(config.gradient_accumulation, 4)
        self.assertTrue(config.bf16)
        for overrides in (
            {"stage": "c"},
            {"model": "large"},
            {"max_steps": 0},
            {"micro_batch_size": 0},
            {"gradient_accumulation": True},
            {"lr": 0.0},
            {"warmup_ratio": 1.1},
            {"grad_clip": float("inf")},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    RuntimeConfig(stage="a", **overrides)

    def test_device_resolution_and_scheduler_curve(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "not available"):
            resolve_device("cuda")
        self.assertEqual(
            warmup_cosine_multiplier(0, warmup_steps=2, total_steps=6), 0.5
        )
        self.assertEqual(
            warmup_cosine_multiplier(1, warmup_steps=2, total_steps=6), 1.0
        )
        self.assertAlmostEqual(
            warmup_cosine_multiplier(5, warmup_steps=2, total_steps=6), 0.0
        )

    def test_auto_micro_batch_probe_descends_only_on_cuda_oom(self) -> None:
        attempted = []

        def probe(size: int) -> None:
            attempted.append(size)
            if size > 3:
                raise torch.OutOfMemoryError("oom")

        selected = auto_micro_batch_probe(5, probe, device=torch.device("cuda"))
        self.assertEqual(selected, 3)
        self.assertEqual(attempted, [5, 4, 3])
        with self.assertRaisesRegex(ValueError, "CUDA"):
            auto_micro_batch_probe(2, probe, device=torch.device("cpu"))


class RunTrainingTests(unittest.TestCase):
    def test_two_optimizer_steps_run_for_stage_a_and_b(self) -> None:
        sample = make_sample()
        for stage, module in (
            ("a", StageAModule(FactorPanelEncoder(ModelConfig.tiny()))),
            (
                "b",
                StageBModule(
                    FactorPanelEncoder(ModelConfig.tiny()),
                    StageBConfig(initial_freeze_steps=0),
                ),
            ),
        ):
            with self.subTest(stage=stage):
                summary = run_training(
                    module,
                    [sample],
                    RuntimeConfig(
                        stage=stage,
                        max_steps=2,
                        gradient_accumulation=2,
                        device="cpu",
                        bf16=False,
                        seed=17,
                    ),
                )
                self.assertEqual(summary.step, 2)
                self.assertEqual(summary.micro_steps, 4)
                self.assertEqual(summary.device, "cpu")
                self.assertTrue(summary.final_loss >= 0.0)
                self.assertGreater(summary.param_count, 0)
                self.assertFalse(summary.bf16_enabled)

    def test_checkpoint_resume_and_stage_mismatch(self) -> None:
        sample = make_sample()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "runtime.pt"
            first_module = StageAModule(FactorPanelEncoder(ModelConfig.tiny()))
            first = run_training(
                first_module,
                [sample],
                RuntimeConfig(
                    stage="a",
                    max_steps=1,
                    gradient_accumulation=1,
                    device="cpu",
                    bf16=False,
                    checkpoint_every=1,
                ),
                checkpoint_path=checkpoint,
            )
            self.assertEqual(first.step, 1)
            self.assertTrue(checkpoint.is_file())

            resumed_module = StageAModule(FactorPanelEncoder(ModelConfig.tiny()))
            resumed = run_training(
                resumed_module,
                [sample],
                RuntimeConfig(
                    stage="a",
                    max_steps=2,
                    gradient_accumulation=1,
                    device="cpu",
                    bf16=False,
                    checkpoint_every=1,
                ),
                checkpoint_path=checkpoint,
                resume=True,
            )
            self.assertEqual(resumed.start_step, 1)
            self.assertEqual(resumed.step, 2)

            stage_b = StageBModule(
                FactorPanelEncoder(ModelConfig.tiny()),
                StageBConfig(initial_freeze_steps=0),
            )
            with self.assertRaisesRegex(ValueError, "stage"):
                run_training(
                    stage_b,
                    [sample],
                    RuntimeConfig(
                        stage="b",
                        max_steps=2,
                        gradient_accumulation=1,
                        device="cpu",
                        bf16=False,
                    ),
                    checkpoint_path=checkpoint,
                    resume=True,
                )

    def test_empty_data_and_invalid_resume_are_rejected(self) -> None:
        module = StageAModule(FactorPanelEncoder(ModelConfig.tiny()))
        config = RuntimeConfig(stage="a", max_steps=1, device="cpu", bf16=False)
        with self.assertRaisesRegex(ValueError, "empty"):
            run_training(module, [], config)
        with self.assertRaisesRegex(ValueError, "checkpoint_path"):
            run_training(module, [make_sample()], config, resume=True)


if __name__ == "__main__":
    unittest.main()
