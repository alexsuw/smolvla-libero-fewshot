"""Naive target-only continuation from the frozen seen checkpoint.

No LoRA, no seen replay, no target-success early stopping. Grid:
3 tasks × {5,10,25} × seeds {42,123}.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vla_fewshot.config import TrainConfig, load_config
from vla_fewshot.data.gates import maybe_assert_no_leakage
from vla_fewshot.data.layout import resolve_datasets_dir
from vla_fewshot.data.leakage import LeakageError
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.storage.sync import execute_local_mirror
from vla_fewshot.training.baseline import (
    TARGET_SLUGS,
    TRAIN_SEEDS,
    apply_cell_overrides,
    assert_baseline_train_config,
    baseline_command,
    baseline_grid,
    build_target_run_id,
    episode_ids_for_cell,
    require_frozen_seen_origin,
)
from vla_fewshot.training.full import require_full_training_runtime
from vla_fewshot.training.full_loop import prepare_full_training, run_full_training
from vla_fewshot.training.resume import assert_override_allowlist
from vla_fewshot.training.trainer import TrainError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/target_baseline.yaml"),
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )
    parser.add_argument("--task", choices=TARGET_SLUGS)
    parser.add_argument("--n-demos", type=int, choices=(5, 10, 25))
    parser.add_argument("--seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--seen-checkpoint", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--profile",
        choices=("static", "full"),
        default="full",
        help="full: SmolVLA on CUDA from the frozen seen checkpoint. "
        "static is not a target trainer.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        help="Allowed resume override: durable mirror destination.",
    )
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="Print the 18 independent baseline commands and exit.",
    )
    return parser


def _load_train_config(path: Path) -> TrainConfig:
    loaded = load_config(path)
    if not isinstance(loaded, TrainConfig):
        raise TypeError(f"{path} is not a train config")
    return loaded


def _full_run_dir(
    args: argparse.Namespace,
    config: TrainConfig,
    *,
    project_root: Path,
    task: str,
    n_demos: int,
) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    from vla_fewshot.paths import resolve_paths

    try:
        paths = resolve_paths()
    except RuntimeError as error:
        raise SystemExit(
            f"{error}; set --output-dir or VLA_DATA_ROOT / VLA_RUNS_DIR. "
            "no GPU training was started."
        ) from error
    return paths.runs_dir / build_target_run_id(
        config, task_slug=task, n_demos=n_demos, project_root=project_root
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if args.print_grid:
        for task, n_demos, seed in baseline_grid():
            print(" ".join(baseline_command(task=task, n_demos=n_demos, seed=seed, config=args.config)))
        return 0

    if args.task is None or args.n_demos is None or args.seed is None:
        parser.error("--task, --n-demos and --seed are required")

    if args.profile == "static":
        print(
            "target baseline has no CPU static trainer; use --profile full on "
            "Linux CUDA after the seen checkpoint is frozen. "
            "no GPU training was started.",
            file=sys.stderr,
        )
        return 1

    try:
        config = _load_train_config(args.config)
        assert_baseline_train_config(config)
        config = apply_cell_overrides(config, seed=args.seed)
        origin, origin_sha256, _run_id = require_frozen_seen_origin(
            checkpoint=args.seen_checkpoint
        )
        splits = load_target_splits(args.split)
        episode_ids = episode_ids_for_cell(
            splits, task_slug=args.task, n_demos=args.n_demos
        )
    except (TrainError, TypeError, ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 1

    try:
        maybe_assert_no_leakage(
            config_path=args.data_config,
            splits_path=args.split,
            output_root=args.output_root,
            stage="target",
            task_slug=args.task,
            extra_episode_ids=episode_ids,
            extra_suite="libero_goal",
            required=True,
        )
    except LeakageError as error:
        print(str(error), file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(f"{error}. no GPU training was started.", file=sys.stderr)
        return 1

    try:
        require_full_training_runtime()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    assert_override_allowlist(
        {
            "log_freq": args.log_freq,
            "destination": args.destination,
            "stop_after": args.stop_after,
            "backup_dir": args.backup_dir,
            "output_dir": args.output_dir,
        }
    )
    project_root = Path.cwd()
    command = ["python", "scripts/train_target.py", *sys.argv[1:]]

    try:
        datasets_dir = resolve_datasets_dir(args.output_root)
        prepared = prepare_full_training(
            config,
            datasets_dir=datasets_dir,
            origin_checkpoint=origin,
            origin_sha256=origin_sha256,
            episode_ids=episode_ids,
        )
        prepared["task_slug"] = args.task
        prepared["task_text"] = splits.tasks[args.task].task_text
        prepared["n_demos"] = args.n_demos
        config = prepared["config"]
        run_dir = _full_run_dir(
            args, config, project_root=project_root, task=args.task, n_demos=args.n_demos
        )
        result = run_full_training(
            config=config,
            run_dir=run_dir,
            command=command,
            config_path=args.config,
            project_root=project_root,
            datasets_dir=datasets_dir,
            resume_from=args.resume_from,
            stop_after=args.stop_after,
            log_freq=args.log_freq,
            install_signal_handlers=True,
            run_id=run_dir.name,
            prepared=prepared,
        )
    except (RuntimeError, FileNotFoundError, FileExistsError, TrainError) as error:
        print(str(error), file=sys.stderr)
        return 1
    mirror = args.backup_dir or args.destination
    if mirror is not None and result.status in {"completed", "stopped"}:
        execute_local_mirror(run_dir, mirror, execute=True)
    print(
        f"train status={result.status} step={result.global_step} "
        f"checkpoint={result.final_checkpoint}"
    )
    return 0 if result.status in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
