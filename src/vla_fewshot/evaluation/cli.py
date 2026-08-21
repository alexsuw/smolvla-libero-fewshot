"""Shared evaluation CLI wiring."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

from vla_fewshot.config import EvalConfig, load_config
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.evaluation.full import refuse_full_evaluation
from vla_fewshot.evaluation.runner import (
    run_static_evaluation,
    static_smoke_config,
)

EvalKind = Literal["target", "zero_shot", "language_control", "seen"]


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--profile",
        choices=("static", "full"),
        default="full",
        help="static: toy protocol smoke. full: LIBERO/SmolVLA on CUDA.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )


def load_eval_config(path: Path) -> EvalConfig:
    loaded = load_config(path)
    if not isinstance(loaded, EvalConfig):
        raise TypeError(f"{path} is not an eval config")
    return loaded


def run_eval_cli(kind: EvalKind, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description={
            "target": "Run resumable fixed-seed target rollouts.",
            "zero_shot": "Run resumable fixed-seed target rollouts.",
            "language_control": "Run paired correct/wrong instruction rollouts.",
            "seen": "Evaluate only the fixed seen probe suite.",
        }[kind]
    )
    add_eval_arguments(parser)
    if kind in {"target", "zero_shot", "language_control"}:
        parser.add_argument(
            "--task",
            choices=("drawer_middle", "bowl_stove", "wine_cabinet"),
        )
    else:
        parser.add_argument("--task", default="synthetic_seen")
    if kind == "target":
        parser.add_argument("--n-demos", type=int, choices=(0, 5, 10, 25))
        parser.add_argument("--seed", type=int, choices=(42, 123))
    args = parser.parse_args(argv)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if args.profile == "full":
        try:
            refuse_full_evaluation()
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1
        return 1

    if args.output_dir is None:
        parser.error("--output-dir is required for static evaluation")
    config = load_eval_config(args.config)
    if args.profile == "static":
        config = static_smoke_config(config)

    task = args.task
    if kind in {"target", "language_control"} and not task:
        parser.error("--task is required")
    if kind == "target" and config.stage == "zero_shot":
        n_demos = 0
        train_seed = None
        method: Literal["baseline", "lora", "replay_lora", "seen"] = "seen"
        stage: Literal["zero_shot", "target_eval", "language_control", "seen_probe"] = "zero_shot"
        language = False
    elif kind == "target":
        if args.n_demos is None or args.seed is None:
            parser.error("--n-demos and --seed are required for target evaluation")
        n_demos = args.n_demos
        train_seed = args.seed
        method = "baseline"
        stage = "target_eval"
        language = False
    elif kind == "language_control":
        n_demos = 0
        train_seed = None
        method = "seen"
        stage = "language_control"
        language = True
    else:
        n_demos = None
        train_seed = None
        method = "seen"
        stage = "seen_probe"
        language = False
        task = args.task or "synthetic_seen"

    checkpoint = args.checkpoint or (args.output_dir / "static.ckpt")
    splits = None
    if n_demos not in (None, 0):
        splits = load_target_splits(args.split)
    result = run_static_evaluation(
        config=config,
        output_dir=args.output_dir,
        checkpoint=checkpoint,
        task_slug=task,
        n_demos=n_demos,
        train_seed=train_seed,
        method=method,
        stage=stage,
        project_root=Path.cwd(),
        splits=splits,
        command=["python", f"scripts/eval_{kind}.py", *sys.argv[1:]],
        language_control=language,
    )
    print(
        f"eval complete={result.complete} planned={result.planned} "
        f"written={result.written} skipped={result.skipped}"
    )
    return 0 if result.complete else 1
