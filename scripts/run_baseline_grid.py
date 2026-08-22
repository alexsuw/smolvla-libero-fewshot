"""Queue the 18 independent naive baseline cells. No N=5→10 chaining."""

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

from vla_fewshot.storage.layout import MANIFEST_NAME
from vla_fewshot.training.baseline import baseline_grid
from vla_fewshot.training.checkpoint import load_json


ROOT = Path(__file__).resolve().parents[1]


def cell_name(task: str, n_demos: int, seed: int) -> str:
    return f"{task}_n{n_demos:02d}_s{seed}"


def _manifest_done(run_dir: Path) -> bool:
    path = run_dir / MANIFEST_NAME
    if not path.is_file():
        return False
    payload = load_json(path)
    return payload.get("status") == "completed"


def _train_command(
    *,
    task: str,
    n_demos: int,
    seed: int,
    output_dir: Path,
    batch_size: int,
    fused_adamw: bool,
    compile_model: bool,
    log_freq: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_target.py",
        "--task",
        task,
        "--n-demos",
        str(n_demos),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--log-freq",
        str(log_freq),
    ]
    if fused_adamw:
        command.append("--fused-adamw")
    if compile_model:
        command.append("--compile")
    return command


def _eval_command(
    *,
    task: str,
    n_demos: int,
    seed: int,
    run_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/eval_target.py",
        "--profile",
        "full",
        "--task",
        task,
        "--n-demos",
        str(n_demos),
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--final-only",
        "--skip-videos",
        "--skip-traces",
        "--output-dir",
        str(output_dir),
    ]


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32, choices=(32, 64, 128))
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/mnt/vla/runs/target_baseline"),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/mnt/vla/eval/target_baseline"),
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("/mnt/vla/validation/TODO28/grid_status.json"),
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-concurrency", type=int, default=2, choices=(1, 2, 4, 6))
    args = parser.parse_args()

    env = os.environ.copy()
    runtime = Path("/mnt/vla/bootstrap/20260821T233035Z/runtime.env")
    if runtime.is_file():
        for line in runtime.read_text(encoding="utf-8").splitlines():
            if line.startswith("export ") and "=" in line:
                key, value = line[len("export ") :].split("=", 1)
                env[key] = value
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")

    cells = [
        {
            "task": task,
            "n_demos": n_demos,
            "seed": seed,
            "name": cell_name(task, n_demos, seed),
            "run_dir": args.runs_root / cell_name(task, n_demos, seed),
            "eval_dir": args.eval_root / cell_name(task, n_demos, seed),
        }
        for task, n_demos, seed in baseline_grid()
    ]
    status = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "batch_size": args.batch_size,
        "fused_adamw": args.fused_adamw,
        "compile_model": args.compile,
        "concurrency": args.concurrency,
        "cells": [cell["name"] for cell in cells],
    }
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    args.status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    log_csv = args.status_path.with_name("grid_train.csv")

    if not args.eval_only:
        pending = [cell for cell in cells if not _manifest_done(cell["run_dir"])]
        print(f"train pending {len(pending)}/{len(cells)}", flush=True)
        active: list[tuple[dict[str, object], subprocess.Popen[bytes], float]] = []
        queue = list(pending)
        failures = 0
        while queue or active:
            while queue and len(active) < args.concurrency:
                cell = queue.pop(0)
                if cell["run_dir"].exists() and not _manifest_done(cell["run_dir"]):
                    print(f"skip incomplete existing dir {cell['run_dir']}", flush=True)
                    failures += 1
                    continue
                command = _train_command(
                    task=cell["task"],
                    n_demos=cell["n_demos"],
                    seed=cell["seed"],
                    output_dir=cell["run_dir"],
                    batch_size=args.batch_size,
                    fused_adamw=args.fused_adamw,
                    compile_model=args.compile,
                    log_freq=args.log_freq,
                )
                print("START", " ".join(command), flush=True)
                proc = subprocess.Popen(command, cwd=ROOT, env=env)
                active.append((cell, proc, time.perf_counter()))
            cell, proc, started = active[0]
            code = proc.wait()
            wall = time.perf_counter() - started
            active.pop(0)
            manifest = {}
            if (cell["run_dir"] / MANIFEST_NAME).is_file():
                manifest = load_json(cell["run_dir"] / MANIFEST_NAME)
            metrics_path = cell["run_dir"] / "metrics.csv"
            last = {}
            if metrics_path.is_file():
                rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
                last = rows[-1] if rows else {}
            row = {
                "task": cell["task"],
                "n_demos": cell["n_demos"],
                "train_seed": cell["seed"],
                "batch_size": args.batch_size,
                "epochs": 100,
                "optimizer_steps": last.get("global_step", ""),
                "wall_clock_s": f"{wall:.1f}",
                "samples_per_sec": last.get("samples_per_second", ""),
                "peak_vram": last.get("gpu_memory_reserved_mb", ""),
                "final_loss": last.get("loss", ""),
                "final_checkpoint": manifest.get("final_checkpoint_uri", ""),
                "status": manifest.get("status", f"exit_{code}"),
            }
            _append_row(log_csv, row)
            print("DONE", row, flush=True)
            if code != 0 or manifest.get("status") != "completed":
                failures += 1
        if failures:
            print(f"training failures={failures}", flush=True)
            return 1

    if args.train_only:
        return 0

    eval_pending = [cell for cell in cells if _manifest_done(cell["run_dir"])]
    print(f"eval cells {len(eval_pending)}", flush=True)
    active_e: list[tuple[dict[str, object], subprocess.Popen[bytes]]] = []
    queue_e = list(eval_pending)
    eval_fail = 0
    while queue_e or active_e:
        while queue_e and len(active_e) < args.eval_concurrency:
            cell = queue_e.pop(0)
            command = _eval_command(
                task=cell["task"],
                n_demos=cell["n_demos"],
                seed=cell["seed"],
                run_dir=cell["run_dir"],
                output_dir=cell["eval_dir"],
            )
            print("EVAL", " ".join(command), flush=True)
            active_e.append((cell, subprocess.Popen(command, cwd=ROOT, env=env)))
        cell, proc = active_e.pop(0)
        code = proc.wait()
        print(f"EVAL_DONE {cell['name']} code={code}", flush=True)
        if code != 0:
            eval_fail += 1
    if eval_fail:
        print(f"eval failures={eval_fail}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
