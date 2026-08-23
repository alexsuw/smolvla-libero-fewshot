"""Fail closed unless every complete baseline checkpoint has ≥20 eval rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.evaluation.baseline_eval import (
    BaselineEvalError,
    baseline_eval_commands,
    default_eval_dir,
    discover_baseline_runs,
    verify_baseline_run_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/splits/target_splits.json"),
    )
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--task", choices=("drawer_middle", "bowl_stove", "wine_cabinet"))
    parser.add_argument("--n-demos", type=int, choices=(1, 2, 5, 10, 25))
    parser.add_argument("--seed", type=int, choices=(42, 123))
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--min-rollouts", type=int, default=20)
    parser.add_argument(
        "--method",
        choices=(
            "baseline",
            "lora",
            "replay_lora",
            "frozen_stats",
            "anchored_l2sp",
        ),
        default="baseline",
    )
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="Print 18 eval_target --run-dir commands and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_grid:
        train_config = {
            "lora": Path("configs/train/target_lora.yaml"),
            "replay_lora": Path("configs/train/target_replay_lora.yaml"),
            "frozen_stats": Path("configs/train/target_frozen_stats.yaml"),
            "anchored_l2sp": Path("configs/train/target_anchored_l2sp.yaml"),
        }.get(args.method, Path("configs/train/target_baseline.yaml"))
        for command in baseline_eval_commands(train_config=train_config):
            print(" ".join(command))
        return 0
    splits = load_target_splits(args.split)
    reports: list[dict] = []
    try:
        if args.train_dir is not None:
            if args.eval_dir is None or args.task is None or args.n_demos is None or args.seed is None:
                parser.error("--train-dir requires --eval-dir --task --n-demos --seed")
            reports.append(
                verify_baseline_run_eval(
                    args.train_dir,
                    args.eval_dir,
                    task_slug=args.task,
                    n_demos=args.n_demos,
                    train_seed=args.seed,
                    splits=splits,
                    min_rollouts=args.min_rollouts,
                    method=args.method,
                )
            )
        elif args.train_root is not None:
            if args.eval_root is None:
                parser.error("--train-root requires --eval-root")
            runs = discover_baseline_runs(args.train_root, method=args.method)
            if not runs:
                raise BaselineEvalError(f"no {args.method} training runs under {args.train_root}")
            for run in runs:
                eval_dir = default_eval_dir(
                    args.eval_root,
                    task_slug=str(run["task_slug"]),
                    n_demos=int(run["n_demos"]),
                    train_seed=int(run["train_seed"]),
                    train_dir=run["train_dir"],
                )
                reports.append(
                    verify_baseline_run_eval(
                        run["train_dir"],
                        eval_dir,
                        task_slug=str(run["task_slug"]),
                        n_demos=int(run["n_demos"]),
                        train_seed=int(run["train_seed"]),
                        splits=splits,
                        min_rollouts=args.min_rollouts,
                        method=args.method,
                    )
                )
        else:
            parser.error("pass --print-grid, --train-dir, or --train-root")
    except BaselineEvalError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"n_runs": len(reports), "runs": reports}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
