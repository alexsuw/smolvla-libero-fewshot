"""Queue the corrected 900-rollout seen-retention sweep.

Adapted weights + libero_90 stats + original seed-only seen-probe reset.
Does not retrain, does not rerun frozen 24/30, and does not touch the
overlay 0/900 tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.evaluation.seen_retention_libero90 import (
    SMOKE_CELL,
    SMOKE_PROBE,
    SMOKE_SEED,
    assert_corrected_rollout_record,
    corrected_retention_command,
    load_original_seen_probe_fingerprints,
    verify_retention_grid,
)
from vla_fewshot.logging.manifest import json_load
from vla_fewshot.storage.layout import MANIFEST_NAME
from vla_fewshot.training.trainer import TrainError


ROOT = Path(__file__).resolve().parents[1]
FROZEN_EVAL_ROOTS = (
    Path("/mnt/vla/eval/seen_retention"),
    Path("/mnt/vla/eval/seen_probes__gd4b8fb8"),
    Path("/mnt/vla/eval/target_baseline"),
    Path("/mnt/vla/eval/target_baseline_n12"),
    Path("/mnt/vla/eval/retention_control"),
    Path("/mnt/vla/eval/zero_shot_v2_seen_stats"),
)


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


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    runtime = Path("/mnt/vla/bootstrap/20260821T233035Z/runtime.env")
    if runtime.is_file():
        for line in runtime.read_text(encoding="utf-8").splitlines():
            if line.startswith("export ") and "=" in line:
                key, value = line[len("export ") :].split("=", 1)
                env[key] = value
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    return env


def _run_one(command: list[str], env: dict[str, str]) -> int:
    command = [sys.executable, *command[1:]]
    print("START", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-runs", type=Path, default=Path("/mnt/vla/runs/target_baseline"))
    parser.add_argument("--n12-runs", type=Path, default=Path("/mnt/vla/runs/target_baseline_n12"))
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/mnt/vla/eval/seen_retention_libero90"),
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("/mnt/vla/validation/TODO28_retention_libero90/grid_status.json"),
    )
    parser.add_argument(
        "--frozen-probe-root",
        type=Path,
        default=Path("/mnt/vla/eval/seen_probes__gd4b8fb8"),
    )
    parser.add_argument("--concurrency", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    if args.eval_root.resolve() in {path.resolve() for path in FROZEN_EVAL_ROOTS}:
        print("refusing to write corrected retention into a frozen result root", file=sys.stderr)
        return 1

    from vla_fewshot.calibration import load_calibration

    probes = tuple(load_calibration().seen_probe_slugs)
    env = _runtime_env()

    try:
        verified = verify_retention_grid(
            official_runs=args.official_runs,
            n12_runs=args.n12_runs,
        )
        fingerprints = load_original_seen_probe_fingerprints(args.frozen_probe_root)
    except (TrainError, FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    cells = []
    for item in verified:
        cells.append(
            {
                **item,
                "eval_dir": args.eval_root / str(item["name"]),
                "run_dir": item["run_dir"],
            }
        )

    status = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metric": "weight_forgetting_adapted_plus_libero90",
        "not_this_tree": "/mnt/vla/eval/seen_retention",
        "concurrency": args.concurrency,
        "n_cells": len(cells),
        "probes": list(probes),
        "planned_rollouts": len(cells) * len(probes) * 10,
        "init_state_mode": "original_seen_probe",
        "cells": [
            {
                "name": cell["name"],
                "checkpoint": str(cell["checkpoint"]),
                "weights_sha256": cell["weights_sha256"],
            }
            for cell in cells
        ],
    }
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    args.status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(
        f"verified {len(cells)} adapted finals; planned "
        f"{status['planned_rollouts']} rollouts",
        flush=True,
    )

    if not args.skip_smoke:
        smoke_task, smoke_n, smoke_seed = SMOKE_CELL
        smoke_cell = next(
            cell
            for cell in cells
            if cell["task"] == smoke_task
            and cell["n_demos"] == smoke_n
            and cell["seed"] == smoke_seed
        )
        smoke_dir = args.status_path.parent / "smoke_v2" / str(smoke_cell["name"])
        command = corrected_retention_command(
            task=smoke_task,
            n_demos=smoke_n,
            seed=smoke_seed,
            run_dir=smoke_cell["run_dir"],
            output_dir=smoke_dir,
            probes=(SMOKE_PROBE,),
            seeds=(SMOKE_SEED,),
        )
        print("SMOKE", " ".join(command), flush=True)
        code = _run_one(command, env)
        if code != 0:
            print(f"smoke eval failed with code {code}", file=sys.stderr)
            return code
        from vla_fewshot.storage.layout import step_directory_name

        rollouts = (
            smoke_dir
            / step_directory_name(int(smoke_cell["step"]))
            / SMOKE_PROBE
            / "rollouts.jsonl"
        )
        if not rollouts.is_file():
            print(f"smoke produced no rollouts at {rollouts}", file=sys.stderr)
            return 1
        row = json.loads(rollouts.read_text(encoding="utf-8").splitlines()[0])
        try:
            passed = assert_corrected_rollout_record(
                row,
                original_fingerprints=fingerprints,
                expected_weights=str(smoke_cell["weights_sha256"]),
                probe=SMOKE_PROBE,
                eval_seed=SMOKE_SEED,
            )
        except TrainError as error:
            print(f"SMOKE_FAIL {error}", file=sys.stderr)
            return 1
        smoke_report = {
            "passed": passed,
            "checkpoint": str(smoke_cell["checkpoint"]),
            "weights_sha256": smoke_cell["weights_sha256"],
            "normalization_suite": row.get("normalization_suite"),
            "normalization_stats_sha256": row.get("normalization_stats_sha256"),
            "task_slug": row.get("task_slug"),
            "eval_seed": row.get("eval_seed"),
            "initial_state_fingerprint": row.get("initial_state_fingerprint"),
            "init_state_mode": row.get("init_state_mode"),
            "success": row.get("success"),
        }
        (args.status_path.parent / "smoke.json").write_text(
            json.dumps(smoke_report, indent=2) + "\n", encoding="utf-8"
        )
        print("SMOKE_OK " + " ".join(passed), flush=True)
        if args.smoke_only:
            return 0

    pending = [
        cell for cell in cells if not _cell_done(cell["eval_dir"], probes, int(cell["step"]))
    ]
    print(f"corrected retention pending {len(pending)}/{len(cells)}", flush=True)
    active: list[tuple[dict[str, object], subprocess.Popen[bytes]]] = []
    queue = list(pending)
    failures = 0
    while queue or active:
        while queue and len(active) < args.concurrency:
            cell = queue.pop(0)
            command = corrected_retention_command(
                task=str(cell["task"]),
                n_demos=int(cell["n_demos"]),
                seed=int(cell["seed"]),
                run_dir=cell["run_dir"],  # type: ignore[arg-type]
                output_dir=cell["eval_dir"],  # type: ignore[arg-type]
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
        print(f"corrected retention failures={failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
