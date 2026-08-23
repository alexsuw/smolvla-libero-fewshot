"""Upload naive target finals to Hugging Face and refresh Hub cards.

Does not retrain. Does not upload optimizer.pt, rng.pt, datasets, or secrets.
Default runtime paths are user-environment only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from vla_fewshot.evaluation.seen_retention import FROZEN_SEEN_SHA256, retention_grid
from vla_fewshot.evaluation.seen_retention_libero90 import verify_retention_grid
from vla_fewshot.storage.layout import (
    CHECKPOINT_CHECKSUMS_NAME,
    CHECKPOINT_COMPLETED_NAME,
    CHECKPOINT_WEIGHTS_PT_NAME,
    NORMALIZATION_STATS_NAME,
    RESOLVED_CONFIG_NAME,
    TRAINABLE_PARAMETERS_NAME,
)
from vla_fewshot.training.trainer import TrainError

SEEN_REPO = "alexsuw/smolvla-libero-fewshot-seen-expert-100k"
NAIVE_REPO = "alexsuw/smolvla-libero-fewshot-naive-baseline"
COLLECTION_SLUG = "alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f"
COLLECTION_TITLE = "SmolVLA LIBERO Few-shot"
EXPECTED_OWNER = "alexsuw"
UPLOAD_NAMES = (
    CHECKPOINT_WEIGHTS_PT_NAME,
    CHECKPOINT_COMPLETED_NAME,
    CHECKPOINT_CHECKSUMS_NAME,
    RESOLVED_CONFIG_NAME,
    NORMALIZATION_STATS_NAME,
    TRAINABLE_PARAMETERS_NAME,
)
ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-runs", type=Path, default=Path("/mnt/vla/runs/target_baseline"))
    parser.add_argument("--n12-runs", type=Path, default=Path("/mnt/vla/runs/target_baseline_n12"))
    parser.add_argument("--seen-repo", default=SEEN_REPO)
    parser.add_argument("--naive-repo", default=NAIVE_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--cards-only", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Upload only these cell names (repeatable). Default: all 30.",
    )
    return parser


def _remote_file(api, repo_id: str, path: str):
    from huggingface_hub.errors import HfHubHTTPError

    try:
        infos = api.get_paths_info(repo_id, [path], repo_type="model")
    except HfHubHTTPError:
        return None
    if not infos:
        return None
    return infos[0]


def _remote_sha256(info) -> str | None:
    lfs = getattr(info, "lfs", None)
    if lfs is None:
        return None
    return getattr(lfs, "sha256", None)


def _cell_files(cell: dict) -> list[tuple[str, Path]]:
    checkpoint = Path(cell["checkpoint"])
    run_dir = Path(cell["run_dir"])
    files = []
    for filename in UPLOAD_NAMES:
        local = checkpoint / filename
        if filename == NORMALIZATION_STATS_NAME and not local.is_file():
            local = run_dir / filename
        if not local.is_file():
            raise FileNotFoundError(
                f"{cell['name']}: missing {filename} under {checkpoint} or {run_dir}"
            )
        files.append((filename, local))
    return files


def _cell_complete_on_hub(api, repo_id: str, cell: dict) -> bool:
    prefix = str(cell["name"])
    for filename, local in _cell_files(cell):
        dest = f"{prefix}/{filename}"
        info = _remote_file(api, repo_id, dest)
        if info is None:
            return False
        remote_size = getattr(info, "size", None)
        if remote_size != local.stat().st_size:
            return False
        if filename == CHECKPOINT_WEIGHTS_PT_NAME:
            remote_sha = _remote_sha256(info)
            if remote_sha and remote_sha != cell["weights_sha256"]:
                return False
    return True


def _upload_cell(api, repo_id: str, cell: dict) -> None:
    name = str(cell["name"])
    checkpoint = Path(cell["checkpoint"])
    files = _cell_files(cell)
    in_checkpoint = [filename for filename, path in files if path.parent == checkpoint]
    extras = [(filename, path) for filename, path in files if path.parent != checkpoint]
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint),
        path_in_repo=name,
        repo_type="model",
        allow_patterns=in_checkpoint,
        ignore_patterns=["optimizer.pt", "rng.pt", "rng.json", "train_state.json"],
        commit_message=f"Add naive cell {name}",
    )
    for filename, local in extras:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"{name}/{filename}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {name}/{filename} from run-level sidecar",
        )


def _publish_cards_and_index(api, *, seen_repo: str, naive_repo: str, cells: list[dict]) -> None:
    seen_card = ROOT / "docs" / "hf" / "seen_expert_README.md"
    naive_card = ROOT / "docs" / "hf" / "naive_baseline_README.md"
    api.upload_file(
        path_or_fileobj=str(seen_card),
        path_in_repo="README.md",
        repo_id=seen_repo,
        repo_type="model",
        commit_message="Update seen-expert model card (task, pins, sources).",
    )
    api.upload_file(
        path_or_fileobj=str(naive_card),
        path_in_repo="README.md",
        repo_id=naive_repo,
        repo_type="model",
        commit_message="Add naive few-shot family model card.",
    )
    if not cells or len(cells) != 30:
        return
    index = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "origin_repo": seen_repo,
        "origin_weights_sha256": FROZEN_SEEN_SHA256,
        "dataset_repo_id": "nvidia/LIBERO_LeRobot_v3",
        "dataset_revision": "e5907374380b8f96511957e6ba5582be52a1e179",
        "base_model": "lerobot/smolvla_base",
        "base_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "collection": COLLECTION_SLUG,
        "n_cells": len(cells),
        "cells": [
            {
                "name": cell["name"],
                "task": cell["task"],
                "n_demos": cell["n_demos"],
                "train_seed": cell["seed"],
                "step": cell["step"],
                "weights_sha256": cell["weights_sha256"],
                "hub_prefix": cell["name"],
            }
            for cell in cells
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix="_naive_index.json", delete=False) as handle:
        handle.write(json.dumps(index, indent=2) + "\n")
        index_path = Path(handle.name)
    api.upload_file(
        path_or_fileobj=str(index_path),
        path_in_repo="index.json",
        repo_id=naive_repo,
        repo_type="model",
        commit_message="Add naive cell index with weights SHA-256.",
    )
    index_path.unlink(missing_ok=True)


def _ensure_collection(api, *, seen_repo: str, naive_repo: str, private: bool):
    collection = api.create_collection(
        title=COLLECTION_TITLE,
        namespace=EXPECTED_OWNER,
        description=(
            "Frozen LIBERO-90 SmolVLA origin plus 30 naive few-shot target "
            "fine-tunes. GitHub: alexsuw/smolvla-libero-fewshot"
        ),
        private=private,
        exists_ok=True,
    )
    if collection.slug != COLLECTION_SLUG:
        print(
            f"collection slug {collection.slug!r} != pinned {COLLECTION_SLUG!r}",
            file=sys.stderr,
        )
    for repo_id, note in (
        (seen_repo, "Frozen seen-expert origin (libero_90, 100k)."),
        (naive_repo, "Naive target few-shot family, all demonstration budgets."),
    ):
        api.add_collection_item(
            collection.slug,
            repo_id,
            item_type="model",
            note=note,
            exists_ok=True,
        )
    return collection


def main() -> int:
    args = build_parser().parse_args()
    if args.cards_only:
        args.skip_verify = True
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    if user != EXPECTED_OWNER:
        print(f"refusing to publish as {user!r}; expected {EXPECTED_OWNER}", file=sys.stderr)
        return 1

    cells: list[dict]
    if args.cards_only and args.skip_verify:
        cells = []
    else:
        try:
            cells = verify_retention_grid(
                official_runs=args.official_runs,
                n12_runs=args.n12_runs,
            )
        except (TrainError, FileNotFoundError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1
        if len(cells) != 30 or len(retention_grid()) != 30:
            print("expected 30 naive finals", file=sys.stderr)
            return 1

    if args.only:
        wanted = set(args.only)
        cells = [cell for cell in cells if cell["name"] in wanted]
        missing = wanted - {cell["name"] for cell in cells}
        if missing:
            print(f"unknown --only cells: {sorted(missing)}", file=sys.stderr)
            return 1

    private = bool(args.private)
    api.create_repo(args.seen_repo, repo_type="model", exist_ok=True, private=True)
    api.create_repo(args.naive_repo, repo_type="model", exist_ok=True, private=True)
    _publish_cards_and_index(
        api,
        seen_repo=args.seen_repo,
        naive_repo=args.naive_repo,
        cells=cells,
    )
    collection = _ensure_collection(
        api,
        seen_repo=args.seen_repo,
        naive_repo=args.naive_repo,
        private=private,
    )
    api.update_repo_settings(args.seen_repo, private=private, repo_type="model")
    api.update_repo_settings(args.naive_repo, private=private, repo_type="model")
    print(f"collection https://huggingface.co/collections/{collection.slug}")
    print(f"seen https://huggingface.co/{args.seen_repo}")
    print(f"naive https://huggingface.co/{args.naive_repo}")

    if args.cards_only or args.skip_weights:
        return 0

    if not cells:
        print("no cells to upload (pass without --skip-verify)", file=sys.stderr)
        return 1

    for cell in cells:
        name = str(cell["name"])
        checkpoint = Path(cell["checkpoint"])
        try:
            if _cell_complete_on_hub(api, args.naive_repo, cell):
                print(f"SKIP {name} already on Hub", flush=True)
                continue
            print(f"UPLOAD {name} {checkpoint}", flush=True)
            _upload_cell(api, args.naive_repo, cell)
        except FileNotFoundError as error:
            print(str(error), file=sys.stderr)
            return 1
        print(f"DONE {name}", flush=True)
    print("UPLOAD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
