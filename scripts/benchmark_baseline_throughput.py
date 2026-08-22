"""Deadline throughput bench. Does not start the 18-cell grid."""

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

from vla_fewshot.data.metadata import load_suite_metadata
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.training.baseline import (
    TRAIN_SEEDS,
    cap_optimizer_steps,
    episode_ids_for_cell,
)

ROOT = Path(__file__).resolve().parents[1]
WARMUP_STEPS = 1000
BATCHES = (32, 64, 128)
PROBE_STEPS = 20


def _cell_table(revision_root: Path) -> list[dict[str, object]]:
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    meta = load_suite_metadata(revision_root, "libero_goal")
    by_id = {int(row["episode_index"]): row for row in meta.episodes}
    rows: list[dict[str, object]] = []
    for slug in splits.tasks:
        for n_demos in (5, 10, 25):
            ids = episode_ids_for_cell(splits, task_slug=slug, n_demos=n_demos)
            frames = sum(int(by_id[item]["length"]) for item in ids)
            for batch in BATCHES:
                stop = cap_optimizer_steps(
                    max_steps=12000,
                    epochs=100,
                    n_samples=frames,
                    effective_batch_size=batch,
                )
                rows.append(
                    {
                        "task": slug,
                        "n_demos": n_demos,
                        "frames": frames,
                        "batch": batch,
                        "stop_at": stop,
                        "warmup_exceeds_run": stop < WARMUP_STEPS,
                    }
                )
    return rows


def _grid_steps(rows: list[dict[str, object]], batch: int) -> int:
    return sum(int(row["stop_at"]) for row in rows if row["batch"] == batch) * len(TRAIN_SEEDS)


def _batch_safe(rows: list[dict[str, object]], batch: int) -> bool:
    return not any(
        row["batch"] == batch and row["warmup_exceeds_run"] for row in rows
    )


def _run(command: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, check=False)


def _metrics_summary(metrics_csv: Path) -> dict[str, float]:
    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    if len(rows) < 5:
        raise RuntimeError(f"not enough metric rows in {metrics_csv}")
    warm = rows[4:]
    step = [float(row["step_time_seconds"]) for row in warm]
    data = [float(row["data_time_seconds"]) for row in warm]
    sps = [float(row["samples_per_second"]) for row in warm]
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    reserved = [float(row["gpu_memory_reserved_mb"]) for row in rows]
    wall = (elapsed[-1] - elapsed[4]) / max(1, len(warm) - 1)
    return {
        "step_time_s": sum(step) / len(step),
        "data_time_s": sum(data) / len(data),
        "samples_per_sec": sum(sps) / len(sps),
        "wall_s_per_step": wall,
        "peak_vram_mb": max(reserved),
        "elapsed_s": elapsed[-1],
    }


def _train_probe(
    *,
    output: Path,
    batch: int | None,
    fused: bool,
    compile_model: bool,
    seed: int,
    env: dict[str, str],
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    command = [
        sys.executable,
        "scripts/train_target.py",
        "--task",
        "drawer_middle",
        "--n-demos",
        "5",
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
        "--stop-after",
        str(PROBE_STEPS),
        "--log-freq",
        "1",
    ]
    if batch is not None:
        command.extend(["--batch-size", str(batch)])
    if fused:
        command.append("--fused-adamw")
    if compile_model:
        command.append("--compile")
    started = time.perf_counter()
    completed = _run(command, env=env, cwd=ROOT)
    wall = time.perf_counter() - started
    summary = {
        "command": command,
        "returncode": completed.returncode,
        "process_wall_s": wall,
    }
    metrics = output / "metrics.csv"
    if completed.returncode == 0 and metrics.is_file():
        summary.update(_metrics_summary(metrics))
    else:
        summary["stderr_tail"] = (completed.stderr or "")[-2000:]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/vla/validation/TODO28/bench"),
    )
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    revision = Path(
        "/mnt/vla/datasets/nvidia_LIBERO_LeRobot_v3/"
        "e5907374380b8f96511957e6ba5582be52a1e179"
    )
    rows = _cell_table(revision)
    exposure = {
        str(batch): {
            "optimizer_steps_18": _grid_steps(rows, batch),
            "warmup_safe": _batch_safe(rows, batch),
        }
        for batch in BATCHES
    }
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "warmup_steps": WARMUP_STEPS,
        "cells": rows,
        "exposure": exposure,
        "probes": {},
    }

    if args.skip_gpu:
        (args.output_dir / "throughput.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(exposure, indent=2))
        return 0

    env = os.environ.copy()
    runtime = Path("/mnt/vla/bootstrap/20260821T233035Z/runtime.env")
    if runtime.is_file():
        for line in runtime.read_text(encoding="utf-8").splitlines():
            if line.startswith("export ") and "=" in line:
                key, value = line[len("export ") :].split("=", 1)
                env[key] = value
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")

    probes: dict[str, object] = {}
    for batch in BATCHES:
        name = f"train_b{batch}"
        print(f"=== {name} ===", flush=True)
        try:
            probes[name] = _train_probe(
                output=args.output_dir / name,
                batch=batch,
                fused=False,
                compile_model=False,
                seed=42,
                env=env,
            )
        except Exception as error:  # noqa: BLE001
            probes[name] = {"error": str(error)}
        print(json.dumps(probes[name], indent=2), flush=True)

    print("=== train_b32_fused ===", flush=True)
    try:
        probes["train_b32_fused"] = _train_probe(
            output=args.output_dir / "train_b32_fused",
            batch=32,
            fused=True,
            compile_model=False,
            seed=42,
            env=env,
        )
    except Exception as error:  # noqa: BLE001
        probes["train_b32_fused"] = {"error": str(error)}
    print(json.dumps(probes["train_b32_fused"], indent=2), flush=True)

    print("=== train_b32_compile ===", flush=True)
    try:
        probes["train_b32_compile"] = _train_probe(
            output=args.output_dir / "train_b32_compile",
            batch=32,
            fused=False,
            compile_model=True,
            seed=42,
            env=env,
        )
    except Exception as error:  # noqa: BLE001
        probes["train_b32_compile"] = {"error": str(error)}
    print(json.dumps(probes["train_b32_compile"], indent=2), flush=True)

    print("=== train_two_jobs_b32 ===", flush=True)
    left = args.output_dir / "train_two_s42"
    right = args.output_dir / "train_two_s123"
    started = time.perf_counter()
    proc_a = subprocess.Popen(
        [
            sys.executable,
            "scripts/train_target.py",
            "--task",
            "drawer_middle",
            "--n-demos",
            "5",
            "--seed",
            "42",
            "--output-dir",
            str(left),
            "--stop-after",
            str(PROBE_STEPS),
            "--log-freq",
            "1",
        ],
        cwd=ROOT,
        env=env,
    )
    proc_b = subprocess.Popen(
        [
            sys.executable,
            "scripts/train_target.py",
            "--task",
            "drawer_middle",
            "--n-demos",
            "5",
            "--seed",
            "123",
            "--output-dir",
            str(right),
            "--stop-after",
            str(PROBE_STEPS),
            "--log-freq",
            "1",
        ],
        cwd=ROOT,
        env=env,
    )
    code_a = proc_a.wait()
    code_b = proc_b.wait()
    two_wall = time.perf_counter() - started
    two: dict[str, object] = {
        "process_wall_s": two_wall,
        "returncodes": [code_a, code_b],
    }
    if code_a == 0 and code_b == 0:
        left_m = _metrics_summary(left / "metrics.csv")
        right_m = _metrics_summary(right / "metrics.csv")
        two["samples_per_sec_sum"] = left_m["samples_per_sec"] + right_m["samples_per_sec"]
        two["left"] = left_m
        two["right"] = right_m
    probes["train_two_jobs_b32"] = two
    print(json.dumps(two, indent=2), flush=True)

    report["probes"] = probes
    recommendation = _recommend(exposure, probes)
    report["recommendation"] = recommendation
    (args.output_dir / "throughput.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("RECOMMENDATION", json.dumps(recommendation, indent=2), flush=True)
    return 0


def _recommend(
    exposure: dict[str, dict[str, object]], probes: dict[str, object]
) -> dict[str, object]:
    candidates: list[tuple[float, dict[str, object]]] = []
    for batch in BATCHES:
        if not exposure[str(batch)]["warmup_safe"]:
            continue
        probe = probes.get(f"train_b{batch}")
        if not isinstance(probe, dict) or "wall_s_per_step" not in probe:
            continue
        steps = int(exposure[str(batch)]["optimizer_steps_18"])
        wall = steps * float(probe["wall_s_per_step"]) + 18 * 50.0
        candidates.append(
            (
                wall,
                {
                    "batch_size": batch,
                    "fused_adamw": False,
                    "compile_model": False,
                    "projected_train_s": wall,
                    "optimizer_steps_18": steps,
                    "wall_s_per_step": probe["wall_s_per_step"],
                    "samples_per_sec": probe["samples_per_sec"],
                },
            )
        )
    fused = probes.get("train_b32_fused")
    if isinstance(fused, dict) and "wall_s_per_step" in fused and exposure["32"]["warmup_safe"]:
        steps = int(exposure["32"]["optimizer_steps_18"])
        wall = steps * float(fused["wall_s_per_step"]) + 18 * 50.0
        candidates.append(
            (
                wall,
                {
                    "batch_size": 32,
                    "fused_adamw": True,
                    "compile_model": False,
                    "projected_train_s": wall,
                    "optimizer_steps_18": steps,
                    "wall_s_per_step": fused["wall_s_per_step"],
                    "samples_per_sec": fused["samples_per_sec"],
                },
            )
        )
    compile_probe = probes.get("train_b32_compile")
    if (
        isinstance(compile_probe, dict)
        and "wall_s_per_step" in compile_probe
        and exposure["32"]["warmup_safe"]
    ):
        steps = int(exposure["32"]["optimizer_steps_18"])
        wall = steps * float(compile_probe["wall_s_per_step"]) + 18 * 50.0
        candidates.append(
            (
                wall,
                {
                    "batch_size": 32,
                    "fused_adamw": False,
                    "compile_model": True,
                    "projected_train_s": wall,
                    "optimizer_steps_18": steps,
                    "wall_s_per_step": compile_probe["wall_s_per_step"],
                    "samples_per_sec": compile_probe["samples_per_sec"],
                },
            )
        )
    if not candidates:
        chosen = {
            "batch_size": 32,
            "fused_adamw": False,
            "compile_model": False,
            "concurrency": 1,
            "reason": "no successful GPU probe; keep frozen batch 32",
        }
    else:
        chosen = min(candidates, key=lambda item: item[0])[1]
    two = probes.get("train_two_jobs_b32")
    chosen["concurrency"] = 1
    if isinstance(two, dict) and "samples_per_sec_sum" in two and "samples_per_sec" in chosen:
        single = float(chosen["samples_per_sec"])
        pair = float(two["samples_per_sec_sum"])
        chosen["two_job_samples_per_sec_sum"] = pair
        if pair > single * 1.15:
            chosen["concurrency"] = 2
            chosen["projected_train_s"] = float(chosen["projected_train_s"]) * single / pair
    return chosen


if __name__ == "__main__":
    raise SystemExit(main())
