import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader


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
    build_stage_b_optimizer,
    collate_factor_samples,
    resolve_device,
    run_training,
    warmup_cosine_multiplier,
)
from factorpanel_fm.runner import _set_optimizer_step_lrs  # noqa: E402


def make_sample(seed: int = 7, decision_date: int | None = None) -> FactorPanelSample:
    config = ModelConfig.tiny()
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(1, config.context_length, 4, generator=generator)
    return FactorPanelSample(
        factor_id="alpha",
        batch=FactorPanelBatch(
            values=values,
            observed_mask=torch.ones_like(values, dtype=torch.bool),
            asset_ids=torch.arange(4).unsqueeze(0),
            dates=torch.arange(config.context_length).unsqueeze(0),
        ),
        future_factor_targets=torch.randn(1, 4, 2, generator=generator),
        future_factor_mask=torch.ones(1, 4, 2, dtype=torch.bool),
        return_targets=torch.randn(1, 4, 3, generator=generator),
        return_mask=torch.ones(1, 4, 3, dtype=torch.bool),
        decision_date=decision_date or config.context_length - 1,
    )


class InterruptAfter:
    def __init__(self, samples: list[FactorPanelSample], count: int) -> None:
        self.samples = samples
        self.count = count

    def __iter__(self):
        yield from self.samples[: self.count]
        raise RuntimeError("simulated interruption")


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

    def test_stage_b_uses_four_groups_and_composes_freeze_with_schedule(self) -> None:
        module = StageBModule(
            FactorPanelEncoder(ModelConfig.tiny()),
            StageBConfig(initial_freeze_steps=2),
        )
        config = RuntimeConfig(
            stage="b",
            max_steps=4,
            warmup_ratio=0.5,
            device="cpu",
            bf16=False,
        )
        optimizer = build_stage_b_optimizer(
            module, step=0, weight_decay=config.weight_decay
        )
        self.assertEqual(len(optimizer.param_groups), 4)

        _set_optimizer_step_lrs(module, optimizer, config, step=0)
        multiplier = warmup_cosine_multiplier(0, warmup_steps=2, total_steps=4)
        for group in optimizer.param_groups:
            expected = 0.0 if group["stage_b_lower"] else config.lr * multiplier
            self.assertEqual(group["lr"], expected)

        _set_optimizer_step_lrs(module, optimizer, config, step=2)
        multiplier = warmup_cosine_multiplier(2, warmup_steps=2, total_steps=4)
        for group in optimizer.param_groups:
            scale = module.config.unfreeze_lr_scale if group["stage_b_lower"] else 1.0
            self.assertEqual(group["lr"], config.lr * scale * multiplier)

    def test_stage_b_runtime_rejects_base_lr_mismatch(self) -> None:
        module = StageBModule(
            FactorPanelEncoder(ModelConfig.tiny()),
            StageBConfig(base_lr=1e-3),
        )
        with self.assertRaisesRegex(ValueError, "base_lr"):
            run_training(
                module,
                [make_sample()],
                RuntimeConfig(stage="b", max_steps=1, device="cpu", bf16=False),
            )

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
            self.assertEqual(resumed.micro_steps, 2)

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

    def test_interrupted_resume_matches_continuous_for_both_stages(self) -> None:
        samples = [make_sample(31, 16), make_sample(37, 17)]
        runtime_by_stage = {
            "a": RuntimeConfig(
                stage="a",
                max_steps=4,
                gradient_accumulation=1,
                warmup_ratio=0.25,
                checkpoint_every=1,
                device="cpu",
                bf16=False,
                seed=101,
            ),
            "b": RuntimeConfig(
                stage="b",
                max_steps=4,
                gradient_accumulation=1,
                warmup_ratio=0.25,
                checkpoint_every=1,
                device="cpu",
                bf16=False,
                seed=101,
            ),
        }
        for stage, runtime in runtime_by_stage.items():
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(211)
                model_config = ModelConfig.tiny(dropout=0.2)
                prototype = (
                    StageAModule(FactorPanelEncoder(model_config))
                    if stage == "a"
                    else StageBModule(FactorPanelEncoder(model_config))
                )
                initial_state = copy.deepcopy(prototype.state_dict())

                continuous = (
                    StageAModule(FactorPanelEncoder(model_config))
                    if stage == "a"
                    else StageBModule(FactorPanelEncoder(model_config))
                )
                continuous.load_state_dict(initial_state)
                continuous_summary = run_training(continuous, samples, runtime)

                interrupted = (
                    StageAModule(FactorPanelEncoder(model_config))
                    if stage == "a"
                    else StageBModule(FactorPanelEncoder(model_config))
                )
                interrupted.load_state_dict(initial_state)
                checkpoint = Path(directory) / f"{stage}.pt"
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_training(
                        interrupted,
                        InterruptAfter(samples, count=1),
                        runtime,
                        checkpoint_path=checkpoint,
                    )

                payload = torch.load(checkpoint, weights_only=True)
                metadata = payload["metadata"]
                self.assertEqual(metadata["consumed_samples"], 1)
                self.assertEqual(metadata["micro_steps_total"], 1)
                self.assertIn("python_rng_state", metadata)
                self.assertIn("torch_cpu_rng_state", metadata)
                self.assertIn("cuda_rng_states", metadata)
                self.assertIn("mps_rng_state", metadata)

                resumed = (
                    StageAModule(FactorPanelEncoder(model_config))
                    if stage == "a"
                    else StageBModule(FactorPanelEncoder(model_config))
                )
                resumed.load_state_dict(initial_state)
                resumed_summary = run_training(
                    resumed,
                    samples,
                    runtime,
                    checkpoint_path=checkpoint,
                    resume=True,
                )

                self.assertEqual(continuous_summary.micro_steps, 4)
                self.assertEqual(resumed_summary.micro_steps, 4)
                for name, expected in continuous.state_dict().items():
                    torch.testing.assert_close(
                        resumed.state_dict()[name],
                        expected,
                        rtol=0,
                        atol=0,
                        msg=lambda message, name=name: f"{stage}:{name}: {message}",
                    )

                one_shot = iter(samples)
                with self.assertRaisesRegex(ValueError, "one-shot iterator"):
                    run_training(
                        resumed,
                        one_shot,
                        runtime,
                        checkpoint_path=checkpoint,
                        resume=True,
                    )
                with self.assertRaisesRegex(ValueError, "deterministic replay"):
                    run_training(
                        resumed,
                        InterruptAfter(samples, count=1),
                        runtime,
                        checkpoint_path=checkpoint,
                        resume=True,
                    )

    def test_resume_rejects_random_dataloader(self) -> None:
        samples = [make_sample(51, 16), make_sample(53, 17)]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "random-loader.pt"
            run_training(
                StageAModule(FactorPanelEncoder(ModelConfig.tiny())),
                samples,
                RuntimeConfig(
                    stage="a",
                    max_steps=1,
                    gradient_accumulation=1,
                    checkpoint_every=1,
                    device="cpu",
                    bf16=False,
                ),
                checkpoint_path=checkpoint,
            )
            random_loader = DataLoader(
                samples,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_factor_samples,
            )
            with self.assertRaisesRegex(ValueError, "SequentialSampler"):
                run_training(
                    StageAModule(FactorPanelEncoder(ModelConfig.tiny())),
                    random_loader,
                    RuntimeConfig(
                        stage="a",
                        max_steps=2,
                        gradient_accumulation=1,
                        device="cpu",
                        bf16=False,
                    ),
                    checkpoint_path=checkpoint,
                    resume=True,
                )

    def test_sequential_dataloader_resume_matches_continuous(self) -> None:
        samples = [make_sample(61, 16), make_sample(67, 17)]
        runtime = RuntimeConfig(
            stage="a",
            max_steps=4,
            gradient_accumulation=1,
            warmup_ratio=0.25,
            checkpoint_every=1,
            device="cpu",
            bf16=False,
            seed=109,
        )
        torch.manual_seed(223)
        config = ModelConfig.tiny(dropout=0.2)
        initial = copy.deepcopy(StageAModule(FactorPanelEncoder(config)).state_dict())
        continuous = StageAModule(FactorPanelEncoder(config))
        continuous.load_state_dict(initial)
        continuous_loader = DataLoader(
            samples,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_factor_samples,
        )
        run_training(continuous, continuous_loader, runtime)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sequential-loader.pt"
            interrupted = StageAModule(FactorPanelEncoder(config))
            interrupted.load_state_dict(initial)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_training(
                    interrupted,
                    InterruptAfter(samples, count=1),
                    runtime,
                    checkpoint_path=checkpoint,
                )
            resumed = StageAModule(FactorPanelEncoder(config))
            resumed.load_state_dict(initial)
            resume_loader = DataLoader(
                samples,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_factor_samples,
            )
            run_training(
                resumed,
                resume_loader,
                runtime,
                checkpoint_path=checkpoint,
                resume=True,
            )

        for name, expected in continuous.state_dict().items():
            torch.testing.assert_close(
                resumed.state_dict()[name], expected, rtol=0, atol=0
            )

    def test_stage_b_lower_blocks_stay_frozen_then_update_at_boundary(self) -> None:
        samples = [make_sample(41, 16), make_sample(43, 17)]
        config = ModelConfig.tiny(dropout=0.1)
        module = StageBModule(
            FactorPanelEncoder(config),
            StageBConfig(initial_freeze_steps=2),
        )
        lower = list(module.encoder.temporal_blocks[0].parameters())
        initial = [parameter.detach().clone() for parameter in lower]
        runtime = RuntimeConfig(
            stage="b",
            max_steps=4,
            gradient_accumulation=1,
            warmup_ratio=0.0,
            checkpoint_every=1,
            device="cpu",
            bf16=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "freeze.pt"
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_training(
                    module,
                    InterruptAfter(samples, count=2),
                    runtime,
                    checkpoint_path=checkpoint,
                )
            self.assertTrue(
                all(
                    torch.equal(parameter, expected)
                    for parameter, expected in zip(lower, initial)
                )
            )

            run_training(
                module, samples, runtime, checkpoint_path=checkpoint, resume=True
            )

        self.assertTrue(
            any(
                not torch.equal(parameter, expected)
                for parameter, expected in zip(lower, initial)
            )
        )


if __name__ == "__main__":
    unittest.main()
