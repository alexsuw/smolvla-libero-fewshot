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
    overlay_eval_rollouts,
    run_static_evaluation,
    static_smoke_config,
)
from vla_fewshot.evaluation.select import list_complete_checkpoints, parse_checkpoint_steps
from vla_fewshot.storage.layout import step_directory_name

EvalKind = Literal["target", "zero_shot", "language_control", "seen"]
TARGET_SLUGS = ("drawer_middle", "bowl_stove", "wine_cabinet")


def eval_cell_output_dir(
    output_dir: Path,
    *,
    label: str,
    task: str,
    run_dir: bool,
    n_tasks: int,
) -> Path:
    """`--run-dir` cells always live under `step_XXXXXX/<task>`, even for one step."""

    if run_dir:
        return output_dir / label / task
    if n_tasks > 1:
        return output_dir / task
    return output_dir


def add_eval_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: Path | None = None,
) -> None:
    if default_config is None:
        parser.add_argument("--config", type=Path, required=True)
    else:
        parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--profile",
        choices=("static", "full"),
        default="full",
        help="static: toy protocol smoke. full: LIBERO/SmolVLA on CUDA.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Evaluate complete checkpoints under a training run. Optional --steps filters.",
    )
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


def normalization_stats_suite(
    eval_config: EvalConfig,
    train_config: TrainConfig,
) -> str:
    """Use the statistics that trained the evaluated policy.

    A frozen seen policy must never be normalized with held-out target-suite
    statistics. Target checkpoints keep their target training suite here;
    subset-local statistics require checkpoint provenance before live M8 eval.
    """

    if eval_config.stage in {"zero_shot", "language_control"}:
        if train_config.stage != "seen":
            raise RuntimeError(
                f"{eval_config.stage} requires a seen train config for normalization; "
                f"got stage={train_config.stage}"
            )
        if train_config.dataset.suite != eval_config.dataset.suite_seen:
            raise RuntimeError(
                f"{eval_config.stage} normalization suite "
                f"{train_config.dataset.suite!r} != configured seen suite "
                f"{eval_config.dataset.suite_seen!r}"
            )
        return train_config.dataset.suite
    if eval_config.stage == "seen_probe":
        return eval_config.dataset.suite_seen
    return train_config.dataset.suite


def _eval_tasks(kind: EvalKind, args: argparse.Namespace, config: EvalConfig) -> list[str]:
    zero_shot = kind == "zero_shot" or config.stage == "zero_shot"
    if kind == "seen":
        if args.profile == "static":
            return [args.task or "synthetic_seen"]
        probes = list(load_calibration().seen_probe_slugs)
        if args.task:
            if args.task not in probes:
                raise SystemExit(f"--task must be one of {probes} for full seen probes")
            return [args.task]
        return probes
    if args.task:
        return [args.task]
    if kind in {"zero_shot", "language_control"} or config.stage in {
        "zero_shot",
        "language_control",
    }:
        return list(TARGET_SLUGS)
    raise SystemExit("--task is required")


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
    if kind == "language_control":
        n_demos = 0
        train_seed = None
        method = "seen"
        stage = "language_control"
        language = True
    elif kind == "zero_shot" or config.stage == "zero_shot":
        n_demos = 0
        train_seed = None
        method = "seen"
        stage = "zero_shot"
    elif config.stage == "language_control":
        n_demos = 0
        train_seed = None
        method = "seen"
        stage = "language_control"
        language = True
    elif kind == "target":
        if args.n_demos is None or args.seed is None:
            raise SystemExit("--n-demos and --seed are required for target evaluation")
        n_demos = args.n_demos
        train_seed = args.seed
        train = _load_train_config(args.train_config)
        if train.method not in {"baseline", "lora", "replay_lora"}:
            raise SystemExit("target eval --train-config must be baseline, lora, or replay_lora")
        method = train.method
        stage = "target_eval"
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
        normalization_stats_sha256,
        suite_stats,
    )

    train = _load_train_config(args.train_config)
    datasets_dir = resolve_datasets_dir(args.output_root)
    stats_suite = normalization_stats_suite(config, train)
    stats = suite_stats(
        datasets_dir=datasets_dir,
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        suite=stats_suite,
    )
    loaded = load_eval_policy(
        checkpoint=checkpoint,
        repo_id=train.model.repo_id,
        revision=train.model.revision,
        scope=train.trainable_scope,
        stats=stats,
        action_chunk_horizon=config.protocol.action_chunk_horizon,
        train=train,
    )
    return LiveRolloutAdapter(
        policy=loaded["policy"],
        preprocessor=loaded["preprocessor"],
        postprocessor=loaded["postprocessor"],
        device=loaded["device"],
        hard_reset=config.protocol.hard_reset,
        normalization_suite=stats_suite,
        normalization_stats_digest=normalization_stats_sha256(stats),
    )


def _can_reuse_eval_weights(args: argparse.Namespace) -> bool:
    return _load_train_config(args.train_config).method not in {"lora", "replay_lora"}


def run_eval_cli(kind: EvalKind, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description={
            "target": "Run resumable fixed-seed target rollouts.",
            "zero_shot": "Run zero-shot final eval: 3 tasks × ≥20, empty train list.",
            "language_control": "Run paired correct/wrong instruction rollouts on the frozen seen checkpoint.",
            "seen": "Evaluate only the fixed seen probe suite.",
        }[kind]
    )
    default_config = {
        "zero_shot": Path("configs/eval/zero_shot.yaml"),
        "language_control": Path("configs/eval/language_control.yaml"),
        "target": Path("configs/eval/final.yaml"),
    }.get(kind)
    add_eval_arguments(parser, default_config=default_config)
    train_default = (
        Path("configs/train/target_baseline.yaml")
        if kind == "target"
        else Path("configs/train/seen_expert.yaml")
    )
    if kind in {"target", "zero_shot", "language_control"}:
        parser.add_argument("--task", choices=TARGET_SLUGS)
    else:
        parser.add_argument("--task", default=None)
        parser.add_argument(
            "--steps",
            help="Comma-separated checkpoint steps to evaluate under --run-dir.",
        )
        parser.add_argument(
            "--rollouts",
            type=int,
            help="Override protocol.rollouts_per_cell (prefix of tracked eval seeds).",
        )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--train-config", type=Path, default=train_default)
    if kind == "target":
        parser.add_argument("--n-demos", type=int, choices=(0, 5, 10, 25))
        parser.add_argument("--seed", type=int, choices=(42, 123))
    if kind in {"zero_shot", "language_control", "target"}:
        parser.add_argument(
            "--print-grid",
            action="store_true",
            help="Print independent eval commands and exit.",
        )
    args = parser.parse_args(argv)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if getattr(args, "print_grid", False):
        if kind == "zero_shot":
            from vla_fewshot.evaluation.zero_shot import zero_shot_commands

            for command in zero_shot_commands(config=args.config):
                print(" ".join(command))
            return 0
        if kind == "language_control":
            from vla_fewshot.evaluation.language_control import language_control_commands

            for command in language_control_commands(config=args.config):
                print(" ".join(command))
            return 0
        if kind == "target":
            from vla_fewshot.evaluation.baseline_eval import target_eval_commands

            for command in target_eval_commands(
                config=args.config, train_config=args.train_config
            ):
                print(" ".join(command))
            return 0

    if args.profile == "full":
        try:
            require_full_evaluation_runtime()
            if kind != "seen":
                assert_seen_checkpoint_frozen()
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1

    if getattr(args, "steps", None) and args.run_dir is None:
        parser.error("--steps requires --run-dir")

    if args.output_dir is None:
        parser.error("--output-dir is required")

    config = load_eval_config(args.config)
    if args.profile == "static":
        if getattr(args, "rollouts", None) is not None:
            parser.error("--rollouts cannot be combined with --profile static")
        config = static_smoke_config(config)
    elif getattr(args, "rollouts", None) is not None:
        config = overlay_eval_rollouts(config, args.rollouts)

    zero_shot = kind == "zero_shot" or config.stage == "zero_shot"
    language = kind == "language_control" or config.stage == "language_control"
    frozen_origin = zero_shot or language
    if frozen_origin:
        from vla_fewshot.evaluation.protocol import ProtocolError

        source = load_eval_config(args.config) if args.profile == "static" else config
        try:
            if kind == "language_control" or (language and not zero_shot):
                from vla_fewshot.evaluation.language_control import (
                    assert_language_control_config,
                )

                assert_language_control_config(source, profile=args.profile)
            else:
                from vla_fewshot.evaluation.zero_shot import assert_zero_shot_config

                assert_zero_shot_config(source, profile=args.profile)
        except ProtocolError as error:
            print(str(error), file=sys.stderr)
            return 1
        if getattr(args, "n_demos", None) not in (None, 0):
            parser.error(f"{config.stage} forbids --n-demos > 0")
        if args.run_dir is not None:
            parser.error(
                f"{config.stage} evaluates only the frozen seen checkpoint; "
                "do not pass --run-dir"
            )

    if kind == "target" and not frozen_origin and not args.task:
        parser.error("--task is required")

    try:
        tasks = _eval_tasks(kind, args, config)
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
    if args.run_dir is not None:
        if args.checkpoint is not None:
            parser.error("pass either --checkpoint or --run-dir, not both")
        listed = list_complete_checkpoints(args.run_dir)
        if getattr(args, "steps", None):
            try:
                wanted = parse_checkpoint_steps(args.steps)
            except ValueError as error:
                parser.error(str(error))
            have = {step: path for step, path in listed}
            missing = [step for step in wanted if step not in have]
            if missing:
                print(
                    f"requested steps missing complete checkpoints: {missing}",
                    file=sys.stderr,
                )
                return 1
            listed = [(step, have[step]) for step in wanted]
        if not listed:
            print(f"no complete checkpoints under {args.run_dir}", file=sys.stderr)
            return 1
        checkpoints = [(step_directory_name(step), path) for step, path in listed]
    elif frozen_origin:
        from vla_fewshot.evaluation.zero_shot import (
            assert_frozen_checkpoint_hash,
            resolve_frozen_eval_checkpoint,
        )

        purpose = "language control" if language else "zero-shot"
        try:
            origin, expected = resolve_frozen_eval_checkpoint(
                args.checkpoint, purpose=purpose
            )
            if not origin.exists():
                raise FileNotFoundError(
                    f"frozen seen checkpoint missing: {origin}. "
                    "no GPU evaluation was started."
                )
            assert_frozen_checkpoint_hash(origin, expected)
        except (RuntimeError, FileNotFoundError) as error:
            print(str(error), file=sys.stderr)
            return 1
        checkpoints = [("ckpt", origin)]
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --profile full")
        checkpoints = [("ckpt", args.checkpoint)]

    codes: list[int] = []
    adapter = None
    reuse_weights = _can_reuse_eval_weights(args)
    try:
        for label, ckpt in checkpoints:
            if adapter is None:
                adapter = _live_adapter(args, config, ckpt)
            elif reuse_weights:
                adapter.load_checkpoint_weights(ckpt)
            else:
                adapter.close()
                adapter = _live_adapter(args, config, ckpt)
            for task in tasks:
                output = eval_cell_output_dir(
                    args.output_dir,
                    label=label,
                    task=task,
                    run_dir=args.run_dir is not None,
                    n_tasks=len(tasks),
                )
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
