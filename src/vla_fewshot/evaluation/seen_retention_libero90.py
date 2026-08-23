"""Corrected seen-retention: adapted weights + libero_90 suite stats.

This is the weight-forgetting protocol. It is not the overlay 0/900
deployment-normalization sweep. Official target-eval roots stay read-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_fewshot.evaluation.seen_retention import (
    FROZEN_SEEN_SHA256,
    PROBE_SEEDS,
    adapted_run_dir,
    cell_name,
    require_final_checkpoint,
    seen_probe_slugs,
)
from vla_fewshot.logging.manifest import json_load
from vla_fewshot.storage.checksums import sha256_file
from vla_fewshot.storage.layout import (
    CHECKPOINT_CHECKSUMS_NAME,
    CHECKPOINT_COMPLETED_NAME,
    CHECKPOINT_WEIGHTS_PT_NAME,
    MANIFEST_NAME,
)
from vla_fewshot.training.trainer import TrainError

LIBERO90_SUITE_STATS_SHA256 = (
    "b159b6fed3e52edf25bd39b377dd64940221b7a030362daf7f726b1c2ecb30cf"
)
INIT_STATE_MODE = "original_seen_probe"
ORIGINAL_INIT_STATE_IDS_PATH = Path("configs/eval/seen_probe_init_state_ids.json")
SMOKE_CELL = ("drawer_middle", 1, 42)
SMOKE_PROBE = "black_bowl_plate"
SMOKE_SEED = 1000


def load_original_init_state_ids(
    path: Path = ORIGINAL_INIT_STATE_IDS_PATH,
) -> dict[tuple[str, int], int]:
    """Pinned init_state_id table that reproduces the frozen 24/30 states."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = payload.get("ids")
    if not isinstance(ids, dict):
        raise TrainError(f"{path} missing ids")
    mapping: dict[tuple[str, int], int] = {}
    for probe in seen_probe_slugs():
        per_seed = ids.get(probe)
        if not isinstance(per_seed, dict):
            raise TrainError(f"{path} missing probe {probe}")
        for seed in PROBE_SEEDS:
            if str(seed) not in per_seed:
                raise TrainError(f"{path} missing {probe} seed {seed}")
            mapping[(probe, seed)] = int(per_seed[str(seed)])
    if mapping[("black_bowl_plate", 1000)] != 10:
        raise TrainError("black_bowl_plate seed 1000 must pin init_state_id 10")
    return mapping


def load_original_seen_probe_fingerprints(probe_root: Path) -> dict[tuple[str, int], str]:
    """Read the frozen 24/30 initial-state hashes. Does not rerun those rollouts."""

    fingerprints: dict[tuple[str, int], str] = {}
    for probe in seen_probe_slugs():
        path = Path(probe_root) / "step_100000" / probe / "rollouts.jsonl"
        if not path.is_file():
            raise TrainError(f"missing original seen-probe rollouts: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("instruction_condition") not in (None, "correct"):
                continue
            if row.get("checkpoint_sha256") != FROZEN_SEEN_SHA256:
                continue
            key = (str(row["task_slug"]), int(row["eval_seed"]))
            fingerprints[key] = str(row["initial_state_fingerprint"])
    expected = {(probe, seed) for probe in seen_probe_slugs() for seed in PROBE_SEEDS}
    missing = sorted(expected - set(fingerprints))
    if missing:
        raise TrainError(f"original seen-probe fingerprints missing {missing}")
    return fingerprints


def verify_adapted_final(run_dir: Path) -> dict[str, Any]:
    """Fail closed unless path and weights.pt match train manifest / COMPLETED."""

    manifest_path = Path(run_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise TrainError(f"missing train manifest: {manifest_path}")
    manifest = json_load(manifest_path)
    if manifest.get("status") != "completed":
        raise TrainError(f"{run_dir} status={manifest.get('status')!r}, expected completed")
    step, checkpoint = require_final_checkpoint(run_dir)
    expected_uri = manifest.get("final_checkpoint_uri")
    if not expected_uri:
        raise TrainError(f"{run_dir} manifest lacks final_checkpoint_uri")
    if checkpoint.resolve() != Path(expected_uri).resolve():
        raise TrainError(
            f"final checkpoint {checkpoint} != manifest {expected_uri}"
        )
    completed_path = checkpoint / CHECKPOINT_COMPLETED_NAME
    checksums_path = checkpoint / CHECKPOINT_CHECKSUMS_NAME
    weights_path = checkpoint / CHECKPOINT_WEIGHTS_PT_NAME
    if not completed_path.is_file() or not checksums_path.is_file() or not weights_path.is_file():
        raise TrainError(f"incomplete checkpoint files under {checkpoint}")
    completed = json_load(completed_path)
    checksums = json_load(checksums_path)
    digest = sha256_file(weights_path)
    expected_completed = completed.get("weights_sha256")
    expected_checksum = (checksums.get("files") or {}).get(CHECKPOINT_WEIGHTS_PT_NAME)
    if digest != expected_completed:
        raise TrainError(
            f"{weights_path} sha256 {digest} != COMPLETED {expected_completed}"
        )
    if digest != expected_checksum:
        raise TrainError(
            f"{weights_path} sha256 {digest} != checksums {expected_checksum}"
        )
    if digest == FROZEN_SEEN_SHA256:
        raise TrainError("adapted final unexpectedly hashes to the frozen seen checkpoint")
    return {
        "run_dir": Path(run_dir),
        "step": step,
        "checkpoint": checkpoint,
        "weights_sha256": digest,
        "manifest_uri": str(expected_uri),
        "task_slug": manifest.get("task_slug"),
        "n_demos": manifest.get("n_demos"),
        "train_seed": manifest.get("train_seed"),
    }


def verify_retention_grid(
    *,
    official_runs: Path,
    n12_runs: Path,
) -> list[dict[str, Any]]:
    from vla_fewshot.evaluation.seen_retention import retention_grid

    cells = []
    for task, n_demos, seed in retention_grid():
        run_dir = adapted_run_dir(
            task=task,
            n_demos=n_demos,
            seed=seed,
            official_runs=official_runs,
            n12_runs=n12_runs,
        )
        verified = verify_adapted_final(run_dir)
        if verified["task_slug"] not in (None, task):
            raise TrainError(f"{run_dir} task {verified['task_slug']} != {task}")
        if verified["n_demos"] not in (None, n_demos):
            raise TrainError(f"{run_dir} n_demos {verified['n_demos']} != {n_demos}")
        if verified["train_seed"] not in (None, seed):
            raise TrainError(f"{run_dir} train_seed {verified['train_seed']} != {seed}")
        cells.append(
            {
                "task": task,
                "n_demos": n_demos,
                "seed": seed,
                "name": cell_name(task, n_demos, seed),
                **verified,
            }
        )
    if len(cells) != 30:
        raise TrainError(f"expected 30 adapted finals, got {len(cells)}")
    return cells


def assert_libero90_suite_stats(
    *,
    source: str,
    suite: str,
    digest: str,
) -> None:
    if source != "suite":
        raise TrainError(
            f"corrected retention refuses non-suite stats source {source!r}"
        )
    if suite != "libero_90":
        raise TrainError(
            f"corrected retention requires libero_90 stats, got {suite!r}"
        )
    if digest != LIBERO90_SUITE_STATS_SHA256:
        raise TrainError(
            f"libero_90 stats hash {digest} != pinned {LIBERO90_SUITE_STATS_SHA256}"
        )


def assert_corrected_rollout_record(
    row: dict[str, Any],
    *,
    original_fingerprints: dict[tuple[str, int], str],
    expected_weights: str,
    probe: str | None = None,
    eval_seed: int | None = None,
) -> list[str]:
    """Return the six integrity check names that passed; raise on the first miss."""

    passed: list[str] = []
    weights = str(row.get("checkpoint_sha256") or "")
    if weights != expected_weights or weights == FROZEN_SEEN_SHA256:
        raise TrainError(
            f"loaded weights {weights} != intended adapted {expected_weights}"
        )
    passed.append("loaded_weights_hash")

    suite = row.get("normalization_suite")
    digest = str(row.get("normalization_stats_sha256") or "")
    if suite != "libero_90":
        raise TrainError(f"stats suite {suite!r} is not libero_90")
    if digest != LIBERO90_SUITE_STATS_SHA256:
        raise TrainError(f"stats hash {digest} is not the pinned libero_90 digest")
    passed.append("stats_source_libero_90")

    task = str(row.get("task_slug") or "")
    allowed = set(seen_probe_slugs())
    if task not in allowed:
        raise TrainError(f"task {task!r} is not one of the original seen probes {sorted(allowed)}")
    if probe is not None and task != probe:
        raise TrainError(f"task {task!r} != requested probe {probe!r}")
    passed.append("seen_probe_task")

    seed = int(row["eval_seed"])
    if seed not in PROBE_SEEDS:
        raise TrainError(f"eval seed {seed} is outside 1000-1009")
    if eval_seed is not None and seed != eval_seed:
        raise TrainError(f"eval seed {seed} != requested {eval_seed}")
    passed.append("eval_seed_1000_1009")

    expected_fp = original_fingerprints.get((task, seed))
    got_fp = row.get("initial_state_fingerprint")
    if expected_fp is None:
        raise TrainError(f"no original fingerprint for {task} seed {seed}")
    if got_fp != expected_fp:
        raise TrainError(
            f"initial-state fingerprint {got_fp} != original seen-probe {expected_fp} "
            f"for {task} seed {seed}"
        )
    if row.get("init_state_mode") not in (None, INIT_STATE_MODE):
        raise TrainError(f"init_state_mode {row.get('init_state_mode')!r} != {INIT_STATE_MODE}")
    passed.append("original_fingerprint")

    if suite == "libero_goal" or "overlay" in str(row.get("notes") or "").lower():
        raise TrainError("target-overlay stats appeared on the corrected retention path")
    if digest == LIBERO90_SUITE_STATS_SHA256 and suite == "libero_90":
        passed.append("no_target_overlay")
    return passed


def corrected_retention_command(
    *,
    task: str,
    n_demos: int,
    seed: int,
    run_dir: Path,
    output_dir: Path,
    probes: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> list[str]:
    command = [
        "python",
        "scripts/eval_seen_retention_libero90.py",
        "--task",
        task,
        "--n-demos",
        str(n_demos),
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--skip-videos",
        "--skip-traces",
    ]
    if probes:
        for probe in probes:
            command.extend(["--probe", probe])
    if seeds:
        command.extend(["--seeds", *[str(item) for item in seeds]])
    return command
