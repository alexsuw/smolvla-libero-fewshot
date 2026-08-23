"""Publish the two validated N=1 research families to Hugging Face Hub.

The publisher is fail-closed. It verifies training provenance, checkpoint
checksums, target/retention protocols, and normalization before it creates or
changes a public release. Optimizer/RNG state, datasets, raw rollouts, traces,
videos, and credentials are never uploaded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from vla_fewshot.evaluation.seen_retention import FROZEN_SEEN_SHA256
from vla_fewshot.evaluation.seen_retention_libero90 import (
    LIBERO90_SUITE_STATS_SHA256,
)
from vla_fewshot.storage.checksums import sha256_file
from vla_fewshot.training.stats import load_normalization_stats, stats_digest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OWNER = "alexsuw"
COLLECTION_SLUG = "alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f"
DATASET_REPO = "nvidia/LIBERO_LeRobot_v3"
DATASET_REVISION = "e5907374380b8f96511957e6ba5582be52a1e179"
BASE_MODEL = "lerobot/smolvla_base"
BASE_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
TASK_STEPS = {"drawer_middle": 500, "bowl_stove": 400, "wine_cabinet": 300}
TRAIN_SEEDS = (42, 123)
RETENTION_PROBES = ("black_bowl_plate", "drawer_bowl", "book_caddy")

CHECKPOINT_FILES = (
    "weights.pt",
    "COMPLETED.json",
    "checksums.json",
    "config.resolved.yaml",
    "normalization_stats.json",
    "trainable_parameters.txt",
)
ADAPTER_FILES = ("adapter/adapter_config.json", "adapter/adapter_model.pt")
RUN_FILES = (
    "manifest.json",
    "environment_manifest.json",
    "trainable_scope.json",
    "metrics.csv",
    "events.jsonl",
    "checkpoints.json",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    manifest_method: str
    trainable_parameters: int
    has_adapter: bool


@dataclass(frozen=True)
class FamilySpec:
    name: str
    repo_id: str
    run_root: Path
    target_root: Path
    retention_root: Path
    card: Path
    results_markdown: Path
    methods: tuple[MethodSpec, ...]


FAMILIES = {
    "lora_n1": FamilySpec(
        name="lora_n1",
        repo_id="alexsuw/smolvla-libero-fewshot-lora-n1",
        run_root=Path("/mnt/vla/runs/task2_n1"),
        target_root=Path("/mnt/vla/eval/task2_n1/target"),
        retention_root=Path("/mnt/vla/eval/task2_n1/retention_libero90"),
        card=ROOT / "docs/hf/lora_n1_README.md",
        results_markdown=ROOT / "report/tables/task2_n1_results.md",
        methods=(
            MethodSpec("target_lora", "lora", 4_215_632, True),
            MethodSpec("replay_lora", "replay_lora", 4_215_632, True),
        ),
    ),
    "stability_n1": FamilySpec(
        name="stability_n1",
        repo_id="alexsuw/smolvla-libero-fewshot-stability-n1",
        run_root=Path("/mnt/vla/runs/target_matched_n1"),
        target_root=Path("/mnt/vla/eval/target_matched_n1"),
        retention_root=Path("/mnt/vla/eval/seen_retention_libero90_matched_n1"),
        card=ROOT / "docs/hf/stability_n1_README.md",
        results_markdown=ROOT / "report/tables/frozen_stats_l2sp_n1_results.md",
        methods=(
            MethodSpec("frozen_stats", "frozen_stats", 99_880_992, False),
            MethodSpec("anchored_l2sp", "anchored_l2sp", 99_880_992, False),
        ),
    ),
}


class ReleaseError(RuntimeError):
    """A release invariant does not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing JSONL file: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    _require(all(isinstance(row, dict) for row in rows), f"invalid JSONL rows: {path}")
    return rows


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_git() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    _require(not status.strip(), "refusing HF publish from a dirty Git worktree")


def _first_demo_ids() -> dict[str, int]:
    split = _read_json(ROOT / "configs/splits/target_splits.json")
    _require(split.get("dataset_revision") == DATASET_REVISION, "split revision drift")
    result = {}
    for task in TASK_STEPS:
        ids = split["tasks"][task]["episode_ids_first_25"]
        _require(ids, f"empty split for {task}")
        result[task] = int(ids[0])
    return result


def _verify_config(path: Path, method: MethodSpec, seed: int) -> None:
    config = yaml.safe_load(path.read_text())
    _require(config["method"] == method.manifest_method, f"method drift: {path}")
    _require(config["dataset"]["repo_id"] == DATASET_REPO, f"dataset drift: {path}")
    _require(
        config["dataset"]["revision"] == DATASET_REVISION,
        f"dataset revision drift: {path}",
    )
    _require(config["training"]["seed"] == seed, f"seed drift: {path}")
    if method.name == "replay_lora":
        replay = config.get("replay") or {}
        _require(replay.get("enabled") is True, "Replay-LoRA replay is disabled")
        _require(replay.get("seen_suite") == "libero_90", "Replay-LoRA source drift")
        _require(replay.get("target_fraction") == 0.75, "Replay target fraction drift")
        _require(replay.get("seen_fraction") == 0.25, "Replay seen fraction drift")
    elif method.name == "target_lora":
        _require(not config.get("replay"), "Target-LoRA unexpectedly enables replay")
    else:
        normalization = config.get("normalization") or {}
        _require(
            normalization.get("expected_sha256") == LIBERO90_SUITE_STATS_SHA256,
            f"frozen normalization drift: {path}",
        )
        _require(
            normalization.get("suite") == "libero_90",
            f"stats suite drift: {path}",
        )
    if method.name == "anchored_l2sp":
        l2sp = config.get("l2sp") or {}
        _require(l2sp.get("enabled") is True, "L2-SP is disabled")
        _require(l2sp.get("strength") == 0.01, "L2-SP strength drift")
        _require(l2sp.get("reduction") == "sum", "L2-SP reduction drift")
        _require(l2sp.get("anchor_dtype") == "fp32", "L2-SP dtype drift")
    elif method.name == "frozen_stats":
        _require(not config.get("l2sp"), "Frozen-Stats unexpectedly enables L2-SP")


def _verify_rollouts(
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    eval_seeds: set[int],
    stage: str,
    task: str,
    train_seed: int,
    method: MethodSpec,
    episode_id: int,
    weights_sha256: str,
    normalization_suite: str,
    normalization_sha256: str,
    fingerprints: dict[tuple[str, str, int], str],
) -> int:
    _require(
        len(rows) == expected_count,
        f"{stage}/{method.name}/{task}: row count drift",
    )
    _require(
        {int(row["eval_seed"]) for row in rows} == eval_seeds,
        f"{stage}/{method.name}/{task}: eval seed drift",
    )
    for row in rows:
        _require(row.get("stage") == stage, f"stage drift in {task}")
        _require(row.get("task_slug") == task, f"task drift in {task}")
        _require(row.get("train_seed") == train_seed, f"train seed drift in {task}")
        _require(
            row.get("method") == method.manifest_method,
            f"eval method drift in {task}",
        )
        _require(row.get("n_demos") == 1, f"N drift in {task}")
        _require(row.get("training_episode_ids") == [episode_id], f"demo drift in {task}")
        _require(
            row.get("checkpoint_sha256") == weights_sha256,
            f"weight SHA drift in {task}",
        )
        _require(
            row.get("dataset_revision") == DATASET_REVISION,
            f"eval revision drift in {task}",
        )
        _require(
            row.get("normalization_suite") == normalization_suite,
            f"stats suite drift in {task}",
        )
        _require(
            row.get("normalization_stats_sha256") == normalization_sha256,
            f"stats SHA drift in {task}",
        )
        _require(row.get("success") in (0, 1), f"invalid success value in {task}")
        key = (stage, task, int(row["eval_seed"]))
        fingerprint = str(row.get("initial_state_fingerprint"))
        _require(
            fingerprint.startswith("sha256:"),
            f"missing init fingerprint in {task}",
        )
        previous = fingerprints.setdefault(key, fingerprint)
        _require(previous == fingerprint, f"init-state mismatch for {key}")
    return sum(int(row["success"]) for row in rows)


def _verify_checkpoint_files(
    checkpoint: Path, *, has_adapter: bool
) -> tuple[str, str | None]:
    checksums = _read_json(checkpoint / "checksums.json").get("files", {})
    completed = _read_json(checkpoint / "COMPLETED.json")
    names = CHECKPOINT_FILES + (ADAPTER_FILES if has_adapter else ())
    observed = {}
    for name in names:
        path = checkpoint / name
        _require(path.is_file(), f"missing release file: {path}")
        if name in {"COMPLETED.json", "checksums.json"}:
            continue
        observed[name] = sha256_file(path)
        _require(checksums.get(name) == observed[name], f"checksum mismatch: {path}")
    weights_sha256 = observed["weights.pt"]
    _require(
        completed.get("weights_sha256") == weights_sha256,
        f"completion marker weight SHA mismatch: {checkpoint}",
    )
    return weights_sha256, observed.get("adapter/adapter_model.pt")


def _build_cell(
    family: FamilySpec,
    method: MethodSpec,
    task: str,
    seed: int,
    episode_id: int,
    fingerprints: dict[tuple[str, str, int], str],
) -> dict[str, Any]:
    step = TASK_STEPS[task]
    cell_name = f"{task}_n01_s{seed}"
    run_dir = family.run_root / method.name / cell_name
    checkpoint = run_dir / "checkpoints" / f"step_{step:06d}"
    manifest = _read_json(run_dir / "manifest.json")
    _require(manifest.get("status") == "completed", f"incomplete run: {run_dir}")
    _require(manifest.get("failure") is None, f"failed run: {run_dir}")
    expected_commit = (
        "cebe04d9ab408eaa37c7ab0249d48fe915ae7b36"
        if family.name == "lora_n1"
        else "f77c469c5f90a2b7bba37988b39848b0b7101abe"
    )
    _require(
        manifest.get("git_commit") == expected_commit,
        f"training commit drift: {run_dir}",
    )
    expected_dirty = family.name == "stability_n1"
    _require(
        manifest.get("git_dirty") is expected_dirty,
        f"unexpected training dirty flag: {run_dir}",
    )
    _require(manifest.get("method") == method.manifest_method, f"method drift: {run_dir}")
    _require(manifest.get("task_slug") == task, f"task drift: {run_dir}")
    _require(manifest.get("train_seed") == seed, f"seed drift: {run_dir}")
    _require(manifest.get("n_demos") == 1, f"N drift: {run_dir}")
    _require(manifest.get("episode_ids") == [episode_id], f"demo drift: {run_dir}")
    _require(manifest.get("base_checkpoint_sha256") == FROZEN_SEEN_SHA256, "base SHA drift")
    _require(manifest.get("dataset_repo_id") == DATASET_REPO, "dataset repo drift")
    _require(manifest.get("dataset_revision") == DATASET_REVISION, "dataset revision drift")
    _require(manifest.get("model_revision") == BASE_REVISION, "base revision drift")
    _require(
        manifest.get("trainable_parameter_count") == method.trainable_parameters,
        f"trainable parameter drift: {run_dir}",
    )
    _verify_config(run_dir / "config.resolved.yaml", method, seed)
    weights_sha256, adapter_sha256 = _verify_checkpoint_files(
        checkpoint, has_adapter=method.has_adapter
    )
    normalization_sha256 = stats_digest(
        load_normalization_stats(checkpoint / "normalization_stats.json")
    )
    if family.name == "stability_n1":
        _require(
            normalization_sha256 == LIBERO90_SUITE_STATS_SHA256,
            f"matched checkpoint normalization drift: {checkpoint}",
        )

    target_path = (
        family.target_root
        / method.name
        / cell_name
        / f"step_{step:06d}"
        / task
        / "rollouts.jsonl"
    )
    target_successes = _verify_rollouts(
        _read_jsonl(target_path),
        expected_count=20,
        eval_seeds=set(range(1000, 1020)),
        stage="target_eval",
        task=task,
        train_seed=seed,
        method=method,
        episode_id=episode_id,
        weights_sha256=weights_sha256,
        normalization_suite=(
            "libero_90" if family.name == "stability_n1" else "libero_goal"
        ),
        normalization_sha256=normalization_sha256,
        fingerprints=fingerprints,
    )
    retention_successes = 0
    probe_results = {}
    for probe in RETENTION_PROBES:
        retention_path = (
            family.retention_root
            / method.name
            / cell_name
            / f"step_{step:06d}"
            / probe
            / "rollouts.jsonl"
        )
        successes = _verify_rollouts(
            _read_jsonl(retention_path),
            expected_count=10,
            eval_seeds=set(range(1000, 1010)),
            stage="seen_retention",
            task=probe,
            train_seed=seed,
            method=method,
            episode_id=episode_id,
            weights_sha256=weights_sha256,
            normalization_suite="libero_90",
            normalization_sha256=LIBERO90_SUITE_STATS_SHA256,
            fingerprints=fingerprints,
        )
        probe_results[probe] = {"successes": successes, "rollouts": 10}
        retention_successes += successes

    for filename in RUN_FILES:
        _require(
            (run_dir / filename).is_file(),
            f"missing run metadata: {run_dir / filename}",
        )
    return {
        "name": f"{method.name}/{cell_name}",
        "method": method.name,
        "manifest_method": method.manifest_method,
        "task": task,
        "n_demos": 1,
        "episode_ids": [episode_id],
        "train_seed": seed,
        "step": step,
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "weights_sha256": weights_sha256,
        "adapter_sha256": adapter_sha256,
        "normalization_stats_sha256": normalization_sha256,
        "base_checkpoint_sha256": FROZEN_SEEN_SHA256,
        "trainable_parameters": method.trainable_parameters,
        "training_git_commit": manifest["git_commit"],
        "training_git_dirty": manifest["git_dirty"],
        "target": {"successes": target_successes, "rollouts": 20},
        "retention": {"successes": retention_successes, "rollouts": 30},
        "retention_by_probe": probe_results,
    }


def build_release(family_names: list[str]) -> dict[str, dict[str, Any]]:
    demos = _first_demo_ids()
    fingerprints: dict[tuple[str, str, int], str] = {}
    releases = {}
    all_weight_hashes: set[str] = set()
    for family_name in family_names:
        family = FAMILIES[family_name]
        cells = [
            _build_cell(family, method, task, seed, demos[task], fingerprints)
            for method in family.methods
            for task in TASK_STEPS
            for seed in TRAIN_SEEDS
        ]
        _require(len(cells) == 12, f"expected 12 cells for {family_name}")
        hashes = {cell["weights_sha256"] for cell in cells}
        _require(len(hashes) == 12, f"duplicate weight hashes in {family_name}")
        _require(not (all_weight_hashes & hashes), "weight hash reused between families")
        all_weight_hashes.update(hashes)
        aggregates = {}
        for method in family.methods:
            selected = [cell for cell in cells if cell["method"] == method.name]
            aggregates[method.name] = {
                "target_successes": sum(cell["target"]["successes"] for cell in selected),
                "target_rollouts": 120,
                "retention_successes": sum(
                    cell["retention"]["successes"] for cell in selected
                ),
                "retention_rollouts": 180,
                "trainable_parameters": method.trainable_parameters,
            }
        releases[family_name] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "release_code_commit": _git_head(),
            "integrity_ok": True,
            "repo_id": family.repo_id,
            "collection": COLLECTION_SLUG,
            "origin_repo": "alexsuw/smolvla-libero-fewshot-seen-expert-100k",
            "origin_weights_sha256": FROZEN_SEEN_SHA256,
            "dataset_repo_id": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "normalization_libero90_sha256": LIBERO90_SUITE_STATS_SHA256,
            "n_cells": len(cells),
            "methods": aggregates,
            "cells": cells,
        }
    return releases


def _remote_file(api, repo_id: str, path: str):
    from huggingface_hub.errors import HfHubHTTPError

    try:
        infos = api.get_paths_info(repo_id, [path], repo_type="model")
    except HfHubHTTPError:
        return None
    return infos[0] if infos else None


def _remote_sha256(info) -> str | None:
    lfs = getattr(info, "lfs", None)
    return getattr(lfs, "sha256", None) if lfs is not None else None


def _release_files(cell: dict[str, Any]) -> list[tuple[str, Path, str | None]]:
    prefix = cell["name"]
    checkpoint = Path(cell["checkpoint"])
    run_dir = Path(cell["run_dir"])
    files = [(f"{prefix}/{name}", checkpoint / name, None) for name in CHECKPOINT_FILES]
    if cell["adapter_sha256"]:
        files.extend(
            (f"{prefix}/{name}", checkpoint / name, None) for name in ADAPTER_FILES
        )
    files.extend(
        (f"{prefix}/run/{name}", run_dir / name, None) for name in RUN_FILES
    )
    expected = []
    for destination, path, _ in files:
        digest = None
        if destination.endswith("/weights.pt"):
            digest = cell["weights_sha256"]
        elif destination.endswith("adapter/adapter_model.pt"):
            digest = cell["adapter_sha256"]
        expected.append((destination, path, digest))
    return expected


def _cell_complete_on_hub(api, repo_id: str, cell: dict[str, Any]) -> bool:
    for destination, local, expected_sha in _release_files(cell):
        info = _remote_file(api, repo_id, destination)
        if info is None or getattr(info, "size", None) != local.stat().st_size:
            return False
        remote_sha = _remote_sha256(info)
        if expected_sha and remote_sha and remote_sha != expected_sha:
            return False
    return True


def _upload_cell(api, repo_id: str, cell: dict[str, Any]) -> None:
    checkpoint = Path(cell["checkpoint"])
    prefix = cell["name"]
    allow = list(CHECKPOINT_FILES)
    if cell["adapter_sha256"]:
        allow.extend(ADAPTER_FILES)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=checkpoint,
        path_in_repo=prefix,
        allow_patterns=allow,
        ignore_patterns=["optimizer.pt", "rng.pt", "rng.json", "train_state.json"],
        commit_message=f"Add verified checkpoint {prefix}",
    )
    run_dir = Path(cell["run_dir"])
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=run_dir,
        path_in_repo=f"{prefix}/run",
        allow_patterns=list(RUN_FILES),
        ignore_patterns=["checkpoints/**", "tensorboard/**", "train.log"],
        commit_message=f"Add provenance for {prefix}",
    )


def _upload_metadata(api, family: FamilySpec, release: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="vla_hf_release_") as tmp:
        staging = Path(tmp)
        (staging / "results").mkdir()
        (staging / "README.md").write_text(family.card.read_text())
        (staging / "index.json").write_text(json.dumps(release, indent=2) + "\n")
        (staging / "results/results.md").write_text(
            family.results_markdown.read_text()
        )
        integrity = {
            "schema_version": 1,
            "integrity_ok": True,
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "release_code_commit": release["release_code_commit"],
            "n_cells": release["n_cells"],
            "target_rollouts": sum(
                cell["target"]["rollouts"] for cell in release["cells"]
            ),
            "retention_rollouts": sum(
                cell["retention"]["rollouts"] for cell in release["cells"]
            ),
            "unique_weight_hashes": len(
                {cell["weights_sha256"] for cell in release["cells"]}
            ),
        }
        (staging / "results/integrity.json").write_text(
            json.dumps(integrity, indent=2) + "\n"
        )
        api.upload_folder(
            repo_id=family.repo_id,
            repo_type="model",
            folder_path=staging,
            commit_message="Publish model card, index, results, and integrity record",
        )


def _update_collection(api, *, private: bool) -> None:
    description = (
        "SmolVLA LIBERO seen origin and Naive, LoRA, Replay, Frozen-Stats, "
        "and L2-SP few-shot checkpoints."
    )
    api.update_collection_metadata(
        COLLECTION_SLUG,
        title="SmolVLA LIBERO Few-shot",
        description=description,
        private=private,
    )
    items = (
        ("alexsuw/smolvla-libero-fewshot-seen-expert-100k", "Frozen seen origin."),
        ("alexsuw/smolvla-libero-fewshot-naive-baseline", "Naive N=1/2/5/10/25."),
        (FAMILIES["lora_n1"].repo_id, "Target-LoRA and Replay-LoRA, N=1."),
        (FAMILIES["stability_n1"].repo_id, "Frozen-Stats FT and L2-SP, N=1."),
    )
    for repo_id, note in items:
        api.add_collection_item(
            COLLECTION_SLUG,
            repo_id,
            item_type="model",
            note=note,
            exists_ok=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(FAMILIES),
        help="Family to verify/publish; repeatable. Default: both.",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    family_names = args.family or list(FAMILIES)
    try:
        releases = build_release(family_names)
        summary = {
            name: {
                "repo_id": release["repo_id"],
                "n_cells": release["n_cells"],
                "methods": release["methods"],
                "integrity_ok": release["integrity_ok"],
            }
            for name, release in releases.items()
        }
        print(json.dumps(summary, indent=2), flush=True)
        if args.verify_only:
            return 0
        _require_clean_git()

        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()["name"]
        _require(user == EXPECTED_OWNER, f"refusing to publish as {user!r}")
        for family_name in family_names:
            family = FAMILIES[family_name]
            api.create_repo(
                family.repo_id,
                repo_type="model",
                exist_ok=True,
                private=True,
            )
            api.update_repo_settings(family.repo_id, private=True, repo_type="model")

        for family_name in family_names:
            family = FAMILIES[family_name]
            release = releases[family_name]
            if not args.skip_weights:
                for index, cell in enumerate(release["cells"], 1):
                    if _cell_complete_on_hub(api, family.repo_id, cell):
                        print(f"SKIP {family_name} {index}/12 {cell['name']}", flush=True)
                    else:
                        print(
                            f"UPLOAD {family_name} {index}/12 {cell['name']}",
                            flush=True,
                        )
                        _upload_cell(api, family.repo_id, cell)
                    _require(
                        _cell_complete_on_hub(api, family.repo_id, cell),
                        f"remote verification failed: {family.repo_id}/{cell['name']}",
                    )
                    print(f"VERIFIED {family.repo_id}/{cell['name']}", flush=True)
            else:
                _require(
                    all(
                        _cell_complete_on_hub(api, family.repo_id, cell)
                        for cell in release["cells"]
                    ),
                    f"--skip-weights requested but {family.repo_id} is incomplete",
                )

        for family_name in family_names:
            family = FAMILIES[family_name]
            _upload_metadata(api, family, releases[family_name])
        for family_name in family_names:
            family = FAMILIES[family_name]
            api.update_repo_settings(
                family.repo_id, private=bool(args.private), repo_type="model"
            )
            print(f"PUBLISHED https://huggingface.co/{family.repo_id}", flush=True)
        _update_collection(api, private=bool(args.private))
        print(f"COLLECTION https://huggingface.co/collections/{COLLECTION_SLUG}")
        return 0
    except (OSError, KeyError, ValueError, ReleaseError) as error:
        print(f"release failed closed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
