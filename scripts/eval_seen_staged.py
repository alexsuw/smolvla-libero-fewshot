"""Two-stage seen-probe eval: coarse 20k–100k screen, then top-two 10-seed final."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vla_fewshot.calibration import load_calibration, load_selected_checkpoint
from vla_fewshot.evaluation.cli import run_eval_cli
from vla_fewshot.evaluation.select import (
    STAGE1_PROBE_ROLLOUTS,
    STAGE2_FINALIST_COUNT,
    STAGE2_PROBE_ROLLOUTS,
    STAGED_PROBE_STEPS,
    collect_probe_scores,
    rank_probe_scores,
    select_seen_checkpoint,
)
from vla_fewshot.reporting.seen_probes import export_seen_probe_report
from vla_fewshot.reproducibility import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/seen_probe.yaml"))
    parser.add_argument("--profile", choices=("static", "full"), default="full")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/seen_expert.yaml"))
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON path for stage-1 ranking and stage-2 selection.",
    )
    return parser


def _eval_argv(args: argparse.Namespace, *, steps: list[int], rollouts: int) -> list[str]:
    argv = [
        "--config",
        str(args.config),
        "--profile",
        args.profile,
        "--run-dir",
        str(args.run_dir),
        "--output-dir",
        str(args.output_dir),
        "--train-config",
        str(args.train_config),
        "--steps",
        ",".join(str(step) for step in steps),
        "--rollouts",
        str(rollouts),
    ]
    if args.output_root is not None:
        argv.extend(["--output-root", str(args.output_root)])
    return argv


def _write_probe_report(probe_root: Path) -> None:
    paths = export_seen_probe_report(probe_root)
    print("wrote probe report:", flush=True)
    for key, path in paths.items():
        print(f"  {key}: {path}", flush=True)


def _score_rows(scores: list) -> list[dict[str, object]]:
    return [
        {
            "step": score.step,
            "mean_success": score.mean_success,
            "per_task": score.per_task,
            "n_rollouts": score.n_rollouts,
        }
        for score in scores
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cal = load_calibration()
    selected = load_selected_checkpoint()
    slugs = list(cal.seen_probe_slugs)

    print(
        "stage 1: steps "
        + ",".join(str(step) for step in STAGED_PROBE_STEPS)
        + f" x {len(slugs)} tasks x {STAGE1_PROBE_ROLLOUTS} seeds",
        flush=True,
    )
    code = run_eval_cli("seen", _eval_argv(args, steps=list(STAGED_PROBE_STEPS), rollouts=STAGE1_PROBE_ROLLOUTS))
    _write_probe_report(args.output_dir)
    if code != 0:
        return code
    stage1 = collect_probe_scores(
        run_dir=args.run_dir,
        probe_root=args.output_dir,
        probe_slugs=slugs,
        steps=STAGED_PROBE_STEPS,
        min_rollouts=STAGE1_PROBE_ROLLOUTS,
    )
    if len(stage1) < STAGE2_FINALIST_COUNT:
        print(
            f"stage 1 needs {STAGE2_FINALIST_COUNT} complete checkpoints, got {len(stage1)}",
            file=sys.stderr,
        )
        return 1
    ranked = rank_probe_scores(stage1)
    finalists = ranked[:STAGE2_FINALIST_COUNT]
    finalist_steps = [score.step for score in finalists]
    print("stage 1 ranking (mean across three libero_90 probes):", flush=True)
    for score in ranked:
        print(
            f"  step={score.step} mean={score.mean_success:.4f} per_task={score.per_task}",
            flush=True,
        )
    print(f"stage 2 finalists: {finalist_steps}", flush=True)

    code = run_eval_cli(
        "seen",
        _eval_argv(args, steps=finalist_steps, rollouts=STAGE2_PROBE_ROLLOUTS),
    )
    _write_probe_report(args.output_dir)
    if code != 0:
        return code
    stage2 = collect_probe_scores(
        run_dir=args.run_dir,
        probe_root=args.output_dir,
        probe_slugs=slugs,
        steps=finalist_steps,
        min_rollouts=STAGE2_PROBE_ROLLOUTS,
    )
    if len(stage2) != STAGE2_FINALIST_COUNT:
        print(
            f"stage 2 needs {STAGE2_FINALIST_COUNT} complete 10-seed cells, got {len(stage2)}",
            file=sys.stderr,
        )
        return 1
    result = select_seen_checkpoint(
        stage2,
        probe_slugs=slugs,
        tolerance=selected.tolerance_success,
        fallback_step=selected.fallback_step,
        indistinguishable_fallback=False,
    )
    payload = {
        "stage1_steps": list(STAGED_PROBE_STEPS),
        "stage1_rollouts": STAGE1_PROBE_ROLLOUTS,
        "stage1_ranking": _score_rows(ranked),
        "stage2_finalists": finalist_steps,
        "stage2_rollouts": STAGE2_PROBE_ROLLOUTS,
        "stage2_scores": _score_rows(rank_probe_scores(stage2)),
        "selected_step": result.score.step,
        "best_mean": result.best_mean,
        "band_steps": result.band_steps,
        "used_fallback": result.used_fallback,
        "rule": result.rule,
        "target_tasks_used": False,
    }
    report = args.report or (args.output_dir / "staged_selection.json")
    atomic_write_json(report, payload, overwrite=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {report}", flush=True)
    _write_probe_report(args.output_dir)
    print("dry-run freeze: run select_seen_checkpoint.py --write after inspecting this report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
