"""Queue 30 naive finals × 3 seen probes. Does not retrain or rerun frozen seen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.evaluation.seen_retention import (
    adapted_run_dir,
    cell_name,
    require_final_checkpoint,
    retention_command,
    retention_grid,
)
from vla_fewshot.logging.manifest import json_load
from vla_fewshot.storage.layout import MANIFEST_NAME


ROOT = Path(__file__).resolve().parents[1]


def _cell_done(eval_dir: Path, probes: tuple[str, ...], step: int) -> bool:
    from vla_fewshot.storage.layout import step_directory_name

    label = step_directory_name(step)
    for probe in probes:
        manifest = eval_dir / label / probe / MANIFEST_NAME
        if not manifest.is_file():
            return False
        if json_load(manifest).get("status") != "completed":
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-runs",
        type=Path,
        default=Path("/mnt/vla/runs/target_baseline"),
    )
    parser.add_argument(
        "--n12-runs",
        type=Path,
        default=Path("/mnt/vla/runs/target_baseline_n12"),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/mnt/vla/eval/seen_retention"),
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("/mnt/vla/validation/TODO28_retention/grid_status.json"),
    )
    parser.add_argument("--concurrency", type=int, default=4, choices=(1, 2, 4))
    args = parser.parse_args()

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
    for task, n_demos, seed in retention_grid():
        run_dir = adapted_run_dir(
            task=task,
            n_demos=n_demos,
            seed=seed,
            official_runs=args.official_runs,
            n12_runs=args.n12_runs,
        )
        step, checkpoint = require_final_checkpoint(run_dir)
        cells.append(
            {
                "task": task,
                "n_demos": n_demos,
                "seed": seed,
                "name": cell_name(task, n_demos, seed),
                "run_dir": run_dir,
                "eval_dir": args.eval_root / cell_name(task, n_demos, seed),
                "step": step,
                "checkpoint": str(checkpoint),
            }
        )

    frozen_roots = {
        Path("/mnt/vla/runs/target_baseline"),
        Path("/mnt/vla/eval/target_baseline"),
        Path("/mnt/vla/runs/target_baseline_n12"),
        Path("/mnt/vla/eval/target_baseline_n12"),
        Path("/mnt/vla/eval/seen_probes__gd4b8fb8"),
    }
    if args.eval_root.resolve() in {path.resolve() for path in frozen_roots}:
        print("refusing to write retention eval into a frozen result root", file=sys.stderr)
        return 1

    status = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "concurrency": args.concurrency,
        "n_cells": len(cells),
        "probes": list(probes),
        "planned_rollouts": len(cells) * len(probes) * 10,
        "cells": [cell["name"] for cell in cells],
    }
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    args.status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    pending = [cell for cell in cells if not _cell_done(cell["eval_dir"], probes, cell["step"])]
    print(f"retention pending {len(pending)}/{len(cells)}", flush=True)
    active: list[tuple[dict[str, object], subprocess.Popen[bytes]]] = []
    queue = list(pending)
    failures = 0
    while queue or active:
        while queue and len(active) < args.concurrency:
            cell = queue.pop(0)
            command = retention_command(
                task=cell["task"],
                n_demos=cell["n_demos"],
                seed=cell["seed"],
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
        print(f"retention failures={failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
