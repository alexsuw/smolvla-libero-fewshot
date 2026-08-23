"""Queue the 2×2 retention control. Does not retrain or rerun the 900 sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.evaluation.retention_control import control_command, control_jobs
from vla_fewshot.evaluation.seen_retention import adapted_run_dir
from vla_fewshot.logging.manifest import json_load
from vla_fewshot.storage.layout import MANIFEST_NAME


ROOT = Path(__file__).resolve().parents[1]
FROZEN_EVAL_ROOTS = (
    Path("/mnt/vla/eval/seen_retention"),
    Path("/mnt/vla/eval/seen_probes__gd4b8fb8"),
    Path("/mnt/vla/eval/target_baseline"),
    Path("/mnt/vla/eval/target_baseline_n12"),
)


def _done(eval_dir: Path, probes: tuple[str, ...]) -> bool:
    manifests = list(eval_dir.rglob(MANIFEST_NAME))
    if len(manifests) < len(probes):
        return False
    completed = 0
    for path in manifests:
        if json_load(path).get("status") == "completed":
            completed += 1
    return completed >= len(probes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-runs", type=Path, default=Path("/mnt/vla/runs/target_baseline"))
    parser.add_argument("--n12-runs", type=Path, default=Path("/mnt/vla/runs/target_baseline_n12"))
    parser.add_argument("--eval-root", type=Path, default=Path("/mnt/vla/eval/retention_control"))
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("/mnt/vla/validation/TODO28_retention_control/grid_status.json"),
    )
    parser.add_argument("--concurrency", type=int, default=4, choices=(1, 2, 4))
    args = parser.parse_args()
    if args.eval_root.resolve() in {path.resolve() for path in FROZEN_EVAL_ROOTS}:
        print("refusing to write control eval into a frozen result root", file=sys.stderr)
        return 1

    from vla_fewshot.calibration import load_calibration

    probes = tuple(load_calibration().seen_probe_slugs)
    env = os.environ.copy()
    runtime = Path("/mnt/vla/bootstrap/20260821T233035Z/runtime.env")
    if runtime.is_file():
        for line in runtime.read_text(encoding="utf-8").splitlines():
            if line.startswith("export ") and "=" in line:
                key, value = line[len("export ") :].split("=", 1)
                env[key] = value
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")

    cells = []
    for job in control_jobs():
        run_dir = adapted_run_dir(
            task=str(job["task"]),
            n_demos=int(job["n_demos"]),
            seed=int(job["seed"]),
            official_runs=args.official_runs,
            n12_runs=args.n12_runs,
        )
        cells.append(
            {
                **job,
                "run_dir": run_dir,
                "eval_dir": args.eval_root / str(job["name"]),
            }
        )

    status = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "concurrency": args.concurrency,
        "n_jobs": len(cells),
        "probes": list(probes),
        "planned_new_rollouts": len(cells) * len(probes) * 5,
        "cells": [cell["name"] for cell in cells],
    }
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    args.status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    pending = [cell for cell in cells if not _done(cell["eval_dir"], probes)]
    print(f"control pending {len(pending)}/{len(cells)}", flush=True)
    active: list[tuple[dict[str, object], subprocess.Popen[bytes]]] = []
    queue = list(pending)
    failures = 0
    while queue or active:
        while queue and len(active) < args.concurrency:
            cell = queue.pop(0)
            command = control_command(
                weights=cell["weights"],  # type: ignore[arg-type]
                stats=cell["stats"],  # type: ignore[arg-type]
                task=str(cell["task"]),
                n_demos=int(cell["n_demos"]),
                seed=int(cell["seed"]),
                run_dir=cell["run_dir"],
                output_dir=cell["eval_dir"],
            )
            command[0] = sys.executable
            print("START", " ".join(command), flush=True)
            active.append((cell, subprocess.Popen(command, cwd=ROOT, env=env)))
        cell, proc = active.pop(0)
        code = proc.wait()
        print(f"DONE {cell['name']} code={code}", flush=True)
        if code != 0:
            failures += 1
    if failures:
        print(f"control failures={failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
