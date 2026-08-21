"""Shared evaluation CLI wiring."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Literal

from vla_fewshot.calibration import load_calibration
from vla_fewshot.config import EvalConfig, TrainConfig, load_config
from vla_fewshot.data.layout import resolve_datasets_dir
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.evaluation.full import (
    assert_seen_checkpoint_frozen,
    require_full_evaluation_runtime,
)
from vla_fewshot.evaluation.runner import (
    run_static_evaluation,
    static_smoke_config,
)
from vla_fewshot.evaluation.select import list_complete_checkpoints
from vla_fewshot.storage.layout import step_directory_name

EvalKind = Literal["target", "zero_shot", "language_control", "seen"]
TARGET_SLUGS = ("drawer_middle", "bowl_stove", "wine_cabinet")


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


def _load_train_config(path: Path) -> TrainConfig:
    loaded = load_config(path)
    if not isinstance(loaded, TrainConfig):
        raise TypeError(f"{path} is not a train config")
    return loaded


def _seen_tasks(kind: EvalKind, args: argparse.Namespace) -> list[str]:
    if kind != "seen":
        if not args.task:
            raise SystemExit("--task is required")
        return [args.task]
    if args.profile == "static":
        return [args.task or "synthetic_seen"]
    probes = list(load_calibration().seen_probe_slugs)
    if args.task:
        if args.task not in probes:
            raise SystemExit(f"--task must be one of {probes} for full seen probes")
        return [args.task]
    return probes


def _run_cell(
    *,
    kind: EvalKind,
    args: argparse.Namespace,
    config: EvalConfig,
    checkpoint: Path,
    task: str,
    output_dir: Path,
    execute_rollout: Any | None,
) -> int:
    n_demos: int | None
    train_seed: int | None
    method: Literal["baseline", "lora", "replay_lora", "seen"]
    stage: Literal["zero_shot", "target_eval", "language_control", "seen_probe"]
    language = False
    if kind in {"target", "zero_shot"} and config.stage == "zero_shot":
        n_demos = 0
        train_seed = None
        method = "seen"
        stage = "zero_shot"
    elif kind == "target":
        if args.n_demos is None or args.seed is None:
            raise SystemExit("--n-demos and --seed are required for target evaluation")
        n_demos = args.n_demos
        train_seed = args.seed
        method = "baseline"
        stage = "target_eval"
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

    splits = None
    if n_demos not in (None, 0):
        splits = load_target_splits(args.split)
    result = run_static_evaluation(
        config=config,
        output_dir=output_dir,
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
        execute_rollout=execute_rollout,
    )
    print(
        f"eval task={task} complete={result.complete} planned={result.planned} "
        f"written={result.written} skipped={result.skipped}"
    )
    return 0 if result.complete else 1


def _live_adapter(args: argparse.Namespace, config: EvalConfig, checkpoint: Path) -> Any:
    from vla_fewshot.evaluation.live import (
        LiveRolloutAdapter,
        load_eval_policy,
        suite_stats,
    )

    train = _load_train_config(args.train_config)
    datasets_dir = resolve_datasets_dir(args.output_root)
    stats = suite_stats(
        datasets_dir=datasets_dir,
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        suite=config.dataset.suite_seen if config.stage == "seen_probe" else config.dataset.suite_target,
    )
    loaded = load_eval_policy(
        checkpoint=checkpoint,
        repo_id=train.model.repo_id,
        revision=train.model.revision,
        scope=train.trainable_scope,
        stats=stats,
        action_chunk_horizon=config.protocol.action_chunk_horizon,
    )
    return LiveRolloutAdapter(
        policy=loaded["policy"],
        preprocessor=loaded["preprocessor"],
        device=loaded["device"],
        hard_reset=config.protocol.hard_reset,
    )


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
        parser.add_argument("--task", choices=TARGET_SLUGS)
    else:
        parser.add_argument("--task", default=None)
        parser.add_argument("--run-dir", type=Path, help="Evaluate every complete checkpoint.")
        parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
        parser.add_argument("--output-root", type=Path)
        parser.add_argument(
            "--train-config",
            type=Path,
            default=Path("configs/train/seen_expert.yaml"),
        )
    if kind == "target":
        parser.add_argument("--n-demos", type=int, choices=(0, 5, 10, 25))
        parser.add_argument("--seed", type=int, choices=(42, 123))
    if kind != "seen":
        parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
        parser.add_argument("--output-root", type=Path)
        parser.add_argument(
            "--train-config",
            type=Path,
            default=Path("configs/train/seen_expert.yaml"),
        )
    args = parser.parse_args(argv)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if args.profile == "full":
        try:
            require_full_evaluation_runtime()
            if kind != "seen":
                assert_seen_checkpoint_frozen()
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1

    if args.output_dir is None:
        parser.error("--output-dir is required")

    config = load_eval_config(args.config)
    if args.profile == "static":
        config = static_smoke_config(config)

    if kind in {"target", "language_control"} and not args.task:
        parser.error("--task is required")

    try:
        tasks = _seen_tasks(kind, args)
    except SystemExit as error:
        parser.error(str(error))

    if args.profile == "static":
        checkpoint = args.checkpoint or (args.output_dir / "static.ckpt")
        codes = [
            _run_cell(
                kind=kind,
                args=args,
                config=config,
                checkpoint=checkpoint,
                task=task,
                output_dir=args.output_dir if len(tasks) == 1 else args.output_dir / task,
                execute_rollout=None,
            )
            for task in tasks
        ]
        return 0 if all(code == 0 for code in codes) else 1

    checkpoints: list[tuple[str, Path]]
    if kind == "seen" and getattr(args, "run_dir", None) is not None:
        if args.checkpoint is not None:
            parser.error("pass either --checkpoint or --run-dir, not both")
        listed = list_complete_checkpoints(args.run_dir)
        if not listed:
            print(f"no complete checkpoints under {args.run_dir}", file=sys.stderr)
            return 1
        checkpoints = [(step_directory_name(step), path) for step, path in listed]
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --profile full")
        checkpoints = [("ckpt", args.checkpoint)]

    codes: list[int] = []
    for label, ckpt in checkpoints:
        adapter = None
        try:
            adapter = _live_adapter(args, config, ckpt)
            for task in tasks:
                if len(checkpoints) > 1:
                    output = args.output_dir / label / task
                elif len(tasks) > 1:
                    output = args.output_dir / task
                else:
                    output = args.output_dir
                codes.append(
                    _run_cell(
                        kind=kind,
                        args=args,
                        config=config,
                        checkpoint=ckpt,
                        task=task,
                        output_dir=output,
                        execute_rollout=adapter,
                    )
                )
        except (RuntimeError, FileNotFoundError, FileExistsError, TypeError) as error:
            print(str(error), file=sys.stderr)
            return 1
        finally:
            if adapter is not None:
                adapter.close()
    return 0 if codes and all(code == 0 for code in codes) else 1
