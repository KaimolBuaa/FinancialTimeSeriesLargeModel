"""Command-line smoke and local parquet pilot entrypoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .batch import FactorPanelBatch
from .data import FactorPanelSample, PanelFrameDataset, collate_factor_samples
from .model import FactorPanelEncoder, ModelConfig
from .runner import RuntimeConfig, RunSummary, run_training
from .training import StageAModule, StageBConfig, StageBModule


def _csv_columns(value: str) -> tuple[str, ...]:
    columns = tuple(item.strip() for item in value.split(",") if item.strip())
    if not columns:
        raise argparse.ArgumentTypeError("column list must be nonempty")
    if len(columns) != len(set(columns)):
        raise argparse.ArgumentTypeError("column list must not contain duplicates")
    return columns


def _model_config(name: str, context_length: int | None = None) -> ModelConfig:
    factory = ModelConfig.tiny if name == "tiny" else ModelConfig.small
    overrides = {} if context_length is None else {"context_length": context_length}
    return factory(**overrides)


def _module(stage: str, config: ModelConfig, return_count: int = 3) -> torch.nn.Module:
    encoder = FactorPanelEncoder(config)
    if stage == "a":
        return StageAModule(encoder)
    if return_count <= 0:
        raise ValueError("Stage B requires labels and nonempty --return-columns")
    horizons = (1, 5, 20) if return_count == 3 else tuple(range(1, return_count + 1))
    return StageBModule(
        encoder,
        StageBConfig(horizons=horizons, initial_freeze_steps=0),
    )


def _synthetic_sample(config: ModelConfig, seed: int) -> FactorPanelSample:
    generator = torch.Generator().manual_seed(seed)
    num_assets = 4
    values = torch.randn(
        1,
        config.context_length,
        num_assets,
        generator=generator,
    )
    return FactorPanelSample(
        factor_id="synthetic_factor",
        batch=FactorPanelBatch(
            values=values,
            observed_mask=torch.ones_like(values, dtype=torch.bool),
            asset_ids=torch.arange(num_assets, dtype=torch.int64).unsqueeze(0),
            dates=torch.arange(config.context_length, dtype=torch.int64).unsqueeze(0),
        ),
        future_factor_targets=torch.randn(1, num_assets, 2, generator=generator),
        future_factor_mask=torch.ones(1, num_assets, 2, dtype=torch.bool),
        return_targets=torch.randn(1, num_assets, 3, generator=generator),
        return_mask=torch.ones(1, num_assets, 3, dtype=torch.bool),
        decision_date=config.context_length - 1,
    )


def _run_smoke(arguments: argparse.Namespace) -> RunSummary:
    torch.manual_seed(arguments.seed)
    model_config = ModelConfig.tiny()
    module = _module(arguments.stage, model_config)
    runtime = RuntimeConfig(
        stage=arguments.stage,
        model="tiny",
        max_steps=arguments.steps,
        micro_batch_size=1,
        gradient_accumulation=4,
        seed=arguments.seed,
        bf16=True,
        checkpoint_every=max(1, arguments.steps),
        device=arguments.device,
    )
    return run_training(
        module,
        [_synthetic_sample(model_config, arguments.seed)],
        runtime,
        checkpoint_path=arguments.checkpoint,
    )


def _run_pilot(arguments: argparse.Namespace) -> RunSummary:
    factors_path = Path(arguments.factors)
    if not factors_path.is_file():
        raise ValueError(f"factors parquet does not exist: {factors_path}")
    labels_path = Path(arguments.labels) if arguments.labels is not None else None
    if labels_path is not None and not labels_path.is_file():
        raise ValueError(f"labels parquet does not exist: {labels_path}")
    if arguments.resume and arguments.checkpoint is None:
        raise ValueError("--resume requires --checkpoint")
    if arguments.stage == "b" and (labels_path is None or not arguments.return_columns):
        raise ValueError("Stage B requires --labels and --return-columns")

    torch.manual_seed(arguments.seed)
    model_config = _model_config(arguments.model, arguments.context_length)
    dataset = PanelFrameDataset.from_parquet(
        factors_path,
        labels_path,
        context_length=model_config.context_length,
        factor_columns=arguments.factor_columns,
        return_columns=arguments.return_columns or (),
    )
    if len(dataset) == 0:
        raise ValueError("pilot dataset has no complete context/future windows")
    loader = DataLoader(
        dataset,
        batch_size=arguments.micro_batch,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_factor_samples,
    )
    module = _module(arguments.stage, model_config, len(dataset.return_columns))
    runtime = RuntimeConfig(
        stage=arguments.stage,
        model=arguments.model,
        max_steps=arguments.steps,
        micro_batch_size=arguments.micro_batch,
        gradient_accumulation=arguments.grad_accum,
        seed=arguments.seed,
        bf16=not arguments.no_bf16,
        checkpoint_every=arguments.checkpoint_every,
        device=arguments.device,
    )
    return run_training(
        module,
        loader,
        runtime,
        checkpoint_path=arguments.checkpoint,
        resume=arguments.resume,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m factorpanel_fm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="run a deterministic Tiny training smoke"
    )
    smoke.add_argument("--stage", choices=("a", "b"), required=True)
    smoke.add_argument("--steps", type=int, default=2)
    smoke.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--checkpoint")
    smoke.set_defaults(handler=_run_smoke)

    pilot = subparsers.add_parser("pilot", help="run a bounded local parquet pilot")
    pilot.add_argument("--factors", required=True)
    pilot.add_argument("--labels")
    pilot.add_argument("--factor-columns", type=_csv_columns, required=True)
    pilot.add_argument("--return-columns", type=_csv_columns)
    pilot.add_argument("--stage", choices=("a", "b"), required=True)
    pilot.add_argument("--model", choices=("tiny", "small"), default="small")
    pilot.add_argument("--context-length", type=int)
    pilot.add_argument("--steps", type=int, default=20000)
    pilot.add_argument("--micro-batch", type=int, default=1)
    pilot.add_argument("--grad-accum", type=int, default=4)
    pilot.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--checkpoint")
    pilot.add_argument("--checkpoint-every", type=int, default=1000)
    pilot.add_argument("--resume", action="store_true")
    pilot.add_argument("--no-bf16", action="store_true")
    pilot.set_defaults(handler=_run_pilot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = arguments.handler(arguments)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(asdict(summary), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main"]
