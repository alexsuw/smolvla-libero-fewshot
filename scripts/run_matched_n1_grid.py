"""Run the preregistered matched N=1 frozen-stat experiment end to end.

This launcher never touches or reruns the naive baseline. It trains the two
registered methods, evaluates 20 fixed target rollouts per checkpoint, then
runs the corrected 3-probe x 10-seed libero_90 retention protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from vla_fewshot.calibration import load_selected_checkpoint
from vla_fewshot.evaluation.protocol import FINAL_SEED_VALUES
from vla_fewshot.evaluation.seen_retention import PROBE_SEEDS, seen_probe_slugs
from vla_fewshot.evaluation.seen_retention_libero90 import (
    LIBERO90_SUITE_STATS_SHA256,
    assert_corrected_rollout_record,
    load_original_seen_probe_fingerprints,
    verify_adapted_final,
)
from vla_fewshot.logging.manifest import json_load
from vla_fewshot.reproducibility import atomic_write_json
from vla_fewshot.storage.layout import MANIFEST_NAME
from vla_fewshot.training.anchored import FROZEN_STATS_SHA256
from vla_fewshot.training.baseline import TARGET_SLUGS, TRAIN_SEEDS
from vla_fewshot.training.stats import load_normalization_stats, stats_digest
from vla_fewshot.training.trainer import TrainError

ROOT = Path(__file__).resolve().parents[1]
METHOD_CONFIGS = {
    "frozen_stats": Path("configs/train/target_frozen_stats.yaml"),
    "anchored_l2sp": Path("configs/train/target_anchored_l2sp.yaml"),
}
N_DEMOS = 1
TARGET_ROLLOUTS = 20
RETENTION_ROLLOUTS = 30


def _cell_name(task: str, seed: int) -> str:
    return f"{task}_n01_s{seed}"


def _cells(runs_root: Path, target_root: Path, retention_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "config": config,
            "task": task,
            "seed": seed,
            "name": _cell_name(task, seed),
            "run_dir": runs_root / method / _cell_name(task, seed),
            "target_dir": target_root / method / _cell_name(task, seed),
            "retention_dir": retention_root / method / _cell_name(task, seed),
        }
        for method, config in METHOD_CONFIGS.items()
        for task in TARGET_SLUGS
        for seed in TRAIN_SEEDS
    ]


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime = Path("/mnt/vla/bootstrap/20260821T233035Z/runtime.env")
    if runtime.is_file():
        for line in runtime.read_text(encoding="utf-8").splitlines():
            if line.startswith("export ") and "=" in line:
                key, value = line[len("export ") :].split("=", 1)
                if "$" not in value:
                    env[key] = value
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    return env


def _manifest_complete(run_dir: Path) -> bool:
    path = run_dir / MANIFEST_NAME
    return path.is_file() and json_load(path).get("status") == "completed"


def _verify_train_cell(cell: dict[str, Any]) -> dict[str, Any]:
    verified = verify_adapted_final(cell["run_dir"])
    selected = load_selected_checkpoint()
    expected = {
        "method": cell["method"],
        "task_slug": cell["task"],
        "n_demos": N_DEMOS,
        "train_seed": cell["seed"],
        "base_checkpoint_sha256": selected.sha256,
    }
    for key, value in expected.items():
        if verified.get(key) != value:
            raise TrainError(
                f"{cell['run_dir']} {key}={verified.get(key)!r} != {value!r}"
            )
    manifest = json_load(cell["run_dir"] / MANIFEST_NAME)
    if manifest.get("normalization_stats_sha256") != FROZEN_STATS_SHA256:
        raise TrainError(f"{cell['run_dir']} normalization hash drifted")
    if manifest.get("normalization_stats_suite") != "libero_90":
        raise TrainError(f"{cell['run_dir']} normalization suite drifted")
    if manifest.get("normalization_stats_scope") != "frozen_seen_suite":
        raise TrainError(f"{cell['run_dir']} normalization scope drifted")
    sidecar = cell["run_dir"] / "normalization_stats.json"
    if not sidecar.is_file() or stats_digest(load_normalization_stats(sidecar)) != FROZEN_STATS_SHA256:
        raise TrainError(f"{cell['run_dir']} normalization sidecar is missing or mismatched")
    return verified


def _train_command(cell: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_target.py",
        "--config",
        str(cell["config"]),
        "--task",
        cell["task"],
        "--n-demos",
        "1",
        "--seed",
        str(cell["seed"]),
        "--output-root",
        str(args.datasets_root),
        "--output-dir",
        str(cell["run_dir"]),
        "--batch-size",
        str(args.batch_size),
        "--log-freq",
        str(args.log_freq),
    ]
    if args.fused_adamw:
        command.append("--fused-adamw")
    return command


def _target_command(cell: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/eval_target.py",
        "--profile",
        "full",
        "--train-config",
        str(cell["config"]),
        "--task",
        cell["task"],
        "--n-demos",
        "1",
        "--seed",
        str(cell["seed"]),
        "--run-dir",
        str(cell["run_dir"]),
        "--final-only",
        "--skip-videos",
        "--skip-traces",
        "--output-root",
        str(args.datasets_root),
        "--output-dir",
        str(cell["target_dir"]),
    ]


def _retention_command(cell: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/eval_seen_retention_libero90.py",
        "--task",
        cell["task"],
        "--n-demos",
        "1",
        "--seed",
        str(cell["seed"]),
        "--run-dir",
        str(cell["run_dir"]),
        "--weight-train-config",
        str(cell["config"]),
        "--stats-train-config",
        "configs/train/seen_expert.yaml",
        "--output-root",
        str(args.datasets_root),
        "--output-dir",
        str(cell["retention_dir"]),
    ]


def _run_queue(
    *,
    stage: str,
    cells: list[dict[str, Any]],
    command_for: Callable[[dict[str, Any]], list[str]],
    concurrency: int,
    env: dict[str, str],
    status: dict[str, Any],
    status_path: Path,
    verify: Callable[[dict[str, Any]], Any] | None = None,
) -> None:
    queue = list(cells)
    active: list[tuple[dict[str, Any], subprocess.Popen[bytes], float]] = []

    def _abort_active() -> None:
        for _cell, process, _started in active:
            if process.poll() is None:
                process.terminate()
        for _cell, process, _started in active:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    while queue or active:
        while queue and len(active) < concurrency:
            cell = queue.pop(0)
            command = command_for(cell)
            print(f"{stage.upper()} START {' '.join(command)}", flush=True)
            process = subprocess.Popen(command, cwd=ROOT, env=env)
            active.append((cell, process, time.perf_counter()))
        completed_index = next(
            (index for index, (_, process, _) in enumerate(active) if process.poll() is not None),
            None,
        )
        if completed_index is None:
            time.sleep(0.25)
            continue
        cell, process, started = active.pop(completed_index)
        code = int(process.returncode or 0)
        wall = time.perf_counter() - started
        key = f"{cell['method']}/{cell['name']}"
        status["cells"].setdefault(key, {})[stage] = {
            "returncode": code,
            "wall_clock_s": wall,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(status_path, status, overwrite=status_path.exists())
        print(f"{stage.upper()} DONE {key} code={code} wall={wall:.1f}s", flush=True)
        if code != 0:
            _abort_active()
            raise TrainError(f"{stage} failed for {key} with exit code {code}")
        if verify is not None:
            try:
                verify(cell)
            except Exception:
                _abort_active()
                raise


def _rollouts(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("rollouts.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _verify_target_cell(cell: dict[str, Any]) -> list[dict[str, Any]]:
    verified = _verify_train_cell(cell)
    rows = [
        row
        for row in _rollouts(cell["target_dir"])
        if row.get("stage") == "target_eval"
        and row.get("instruction_condition") in (None, "correct")
    ]
    seeds = sorted(int(row["eval_seed"]) for row in rows)
    if len(rows) != TARGET_ROLLOUTS or seeds != list(FINAL_SEED_VALUES):
        raise TrainError(
            f"{cell['target_dir']} target rollouts/seeds incomplete: {len(rows)} {seeds}"
        )
    for row in rows:
        if row.get("method") != cell["method"] or row.get("task_slug") != cell["task"]:
            raise TrainError(f"{cell['target_dir']} target method/task mismatch")
        if int(row.get("n_demos") or 0) != 1 or int(row.get("train_seed") or -1) != cell["seed"]:
            raise TrainError(f"{cell['target_dir']} target N/seed mismatch")
        if row.get("checkpoint_sha256") != verified["weights_sha256"]:
            raise TrainError(f"{cell['target_dir']} target weights hash mismatch")
        if row.get("normalization_suite") != "libero_90":
            raise TrainError(f"{cell['target_dir']} target stats suite mismatch")
        if row.get("normalization_stats_sha256") != FROZEN_STATS_SHA256:
            raise TrainError(f"{cell['target_dir']} target stats hash mismatch")
    return rows


def _verify_retention_cell(
    cell: dict[str, Any],
    fingerprints: dict[tuple[str, int], str],
) -> list[dict[str, Any]]:
    verified = _verify_train_cell(cell)
    rows = [
        row
        for row in _rollouts(cell["retention_dir"])
        if row.get("stage") == "seen_retention"
        and row.get("instruction_condition") in (None, "correct")
    ]
    expected_pairs = {
        (probe, seed) for probe in seen_probe_slugs() for seed in PROBE_SEEDS
    }
    got_pairs = {(str(row["task_slug"]), int(row["eval_seed"])) for row in rows}
    if len(rows) != RETENTION_ROLLOUTS or got_pairs != expected_pairs:
        raise TrainError(f"{cell['retention_dir']} retention grid is incomplete")
    for row in rows:
        if row.get("method") != cell["method"]:
            raise TrainError(f"{cell['retention_dir']} retention method mismatch")
        assert_corrected_rollout_record(
            row,
            original_fingerprints=fingerprints,
            expected_weights=verified["weights_sha256"],
        )
    return rows


def _metrics(run_dir: Path) -> dict[str, float]:
    rows = list(csv.DictReader((run_dir / "metrics.csv").open(encoding="utf-8")))
    if not rows:
        raise TrainError(f"missing metrics rows under {run_dir}")
    return {
        "train_elapsed_s": float(rows[-1]["elapsed_seconds"]),
        "peak_vram_mb": max(float(row["gpu_memory_reserved_mb"]) for row in rows),
    }


def _summarize(
    cells: list[dict[str, Any]],
    status: dict[str, Any],
    status_path: Path,
    fingerprints: dict[tuple[str, int], str],
) -> dict[str, Any]:
    per_cell: list[dict[str, Any]] = []
    for cell in cells:
        target = _verify_target_cell(cell)
        retention = _verify_retention_cell(cell, fingerprints)
        manifest = json_load(cell["run_dir"] / MANIFEST_NAME)
        metrics = _metrics(cell["run_dir"])
        key = f"{cell['method']}/{cell['name']}"
        train_wall = status["cells"].get(key, {}).get("train", {}).get("wall_clock_s")
        per_cell.append(
            {
                "method": cell["method"],
                "task": cell["task"],
                "train_seed": cell["seed"],
                "target_successes": sum(int(row.get("success") or 0) for row in target),
                "target_rollouts": len(target),
                "retention_successes": sum(
                    int(row.get("success") or 0) for row in retention
                ),
                "retention_rollouts": len(retention),
                "trainable_parameters": manifest.get("trainable_parameter_count"),
                "process_wall_clock_s": train_wall,
                **metrics,
            }
        )
    methods: dict[str, Any] = {}
    for method in METHOD_CONFIGS:
        subset = [row for row in per_cell if row["method"] == method]
        target_successes = sum(row["target_successes"] for row in subset)
        target_rollouts = sum(row["target_rollouts"] for row in subset)
        seen_successes = sum(row["retention_successes"] for row in subset)
        seen_rollouts = sum(row["retention_rollouts"] for row in subset)
        walls = [
            float(row["process_wall_clock_s"])
            for row in subset
            if row["process_wall_clock_s"] is not None
        ]
        methods[method] = {
            "target_successes": target_successes,
            "target_rollouts": target_rollouts,
            "target_rate": target_successes / target_rollouts,
            "retention_successes": seen_successes,
            "retention_rollouts": seen_rollouts,
            "retention_rate": seen_successes / seen_rollouts,
            "trainable_parameters": subset[0]["trainable_parameters"],
            "mean_process_wall_clock_s": sum(walls) / len(walls) if walls else None,
            "mean_train_elapsed_s": sum(row["train_elapsed_s"] for row in subset)
            / len(subset),
            "peak_vram_mb": max(row["peak_vram_mb"] for row in subset),
        }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "integrity_ok": True,
        "normalization_stats_sha256": LIBERO90_SUITE_STATS_SHA256,
        "methods": methods,
        "per_cell": per_cell,
    }
    summary_path = status_path.with_name("summary.json")
    atomic_write_json(summary_path, payload, overwrite=summary_path.exists())
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/mnt/vla/runs/target_matched_n1"),
    )
    parser.add_argument(
        "--target-eval-root",
        type=Path,
        default=Path("/mnt/vla/eval/target_matched_n1"),
    )
    parser.add_argument(
        "--retention-eval-root",
        type=Path,
        default=Path("/mnt/vla/eval/seen_retention_libero90_matched_n1"),
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("/mnt/vla/validation/TODO32/grid_status.json"),
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/mnt/vla/datasets"),
    )
    parser.add_argument("--concurrency", type=int, choices=(2, 4), default=4)
    parser.add_argument("--eval-concurrency", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--batch-size", type=int, choices=(32,), default=32)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "target", "retention", "summarize"),
        default="all",
    )
    parser.add_argument("--print-grid", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = _cells(args.runs_root, args.target_eval_root, args.retention_eval_root)
    if len(cells) != 12 or len({(c["method"], c["task"], c["seed"]) for c in cells}) != 12:
        raise TrainError("matched N=1 grid must contain exactly 12 independent cells")
    if args.print_grid:
        for cell in cells:
            print(" ".join(_train_command(cell, args)))
        return 0

    selected = load_selected_checkpoint()
    if selected.status != "frozen" or not selected.sha256:
        raise TrainError("selected seen checkpoint is not frozen")
    if FROZEN_STATS_SHA256 != LIBERO90_SUITE_STATS_SHA256:
        raise TrainError("training/eval libero_90 stats pins disagree")
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    if args.status_path.is_file():
        status = json_load(args.status_path)
        expected_status = {
            "methods": list(METHOD_CONFIGS),
            "n_demos": 1,
            "train_concurrency": args.concurrency,
            "eval_concurrency": args.eval_concurrency,
            "batch_size": args.batch_size,
            "fused_adamw": args.fused_adamw,
            "seen_checkpoint_sha256": selected.sha256,
            "normalization_stats_sha256": FROZEN_STATS_SHA256,
        }
        for key, expected in expected_status.items():
            if status.get(key) != expected:
                raise TrainError(
                    f"existing pipeline status {key}={status.get(key)!r} "
                    f"!= requested {expected!r}"
                )
    else:
        status = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "methods": list(METHOD_CONFIGS),
            "n_demos": 1,
            "train_concurrency": args.concurrency,
            "eval_concurrency": args.eval_concurrency,
            "batch_size": args.batch_size,
            "fused_adamw": args.fused_adamw,
            "seen_checkpoint_sha256": selected.sha256,
            "normalization_stats_sha256": FROZEN_STATS_SHA256,
            "cells": {},
        }
        atomic_write_json(args.status_path, status)

    env = _runtime_env()
    train_cells = []
    for cell in cells:
        if _manifest_complete(cell["run_dir"]):
            _verify_train_cell(cell)
            continue
        if cell["run_dir"].exists():
            raise TrainError(
                f"incomplete existing run refuses overwrite: {cell['run_dir']}"
            )
        train_cells.append(cell)

    if args.phase in {"all", "train"}:
        _run_queue(
            stage="train",
            cells=train_cells,
            command_for=lambda cell: _train_command(cell, args),
            concurrency=args.concurrency,
            env=env,
            status=status,
            status_path=args.status_path,
            verify=_verify_train_cell,
        )
    if args.phase == "train":
        return 0

    for cell in cells:
        _verify_train_cell(cell)
    if args.phase in {"all", "target"}:
        _run_queue(
            stage="target",
            cells=cells,
            command_for=lambda cell: _target_command(cell, args),
            concurrency=args.eval_concurrency,
            env=env,
            status=status,
            status_path=args.status_path,
            verify=_verify_target_cell,
        )
    if args.phase == "target":
        return 0

    frozen_probe_root = Path("/mnt/vla/eval/seen_probes__gd4b8fb8")
    fingerprints = load_original_seen_probe_fingerprints(frozen_probe_root)
    if args.phase in {"all", "retention"}:
        _run_queue(
            stage="retention",
            cells=cells,
            command_for=lambda cell: _retention_command(cell, args),
            concurrency=args.eval_concurrency,
            env=env,
            status=status,
            status_path=args.status_path,
            verify=lambda cell: _verify_retention_cell(cell, fingerprints),
        )
    if args.phase == "retention":
        return 0

    summary = _summarize(cells, status, args.status_path, fingerprints)
    print(json.dumps(summary["methods"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
