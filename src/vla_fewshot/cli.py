"""Shared fail-safe CLI contracts for not-yet-implemented milestones."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from vla_fewshot.config import load_config


@dataclass(frozen=True)
class CommandSpec:
    description: str
    milestone: str
    arguments: tuple[str, ...] = ()


COMMANDS: dict[str, CommandSpec] = {
    "doctor": CommandSpec("Validate runtime, GPU, EGL, storage, and pins.", "M1", ("config",)),
    "resolve_revisions": CommandSpec("Resolve and verify immutable upstream revisions.", "M1"),
    "download_dataset": CommandSpec(
        "Download pinned metadata or selected dataset artifacts.",
        "M2",
        ("config", "metadata_only", "suite"),
    ),
    "inspect_dataset": CommandSpec(
        "Inspect schema, counts, tasks, episodes, and statistics.", "M2", ("config",)
    ),
    "verify_split": CommandSpec("Verify exact tracked episode prefixes.", "M2", ("split",)),
    "verify_no_leakage": CommandSpec("Fail on target-data or protocol leakage.", "M2"),
    "materialize_subset": CommandSpec(
        "Create an immutable logical or physical episode subset.",
        "M2",
        ("task", "n_demos"),
    ),
    "check_observation_parity": CommandSpec(
        "Save dataset/environment camera parity evidence.", "M3", ("config", "task")
    ),
    "replay_expert": CommandSpec(
        "Replay expert actions through production adapters.",
        "M3",
        ("config", "task", "episode_id", "save_video"),
    ),
    "smoke_inference": CommandSpec(
        "Load pinned SmolVLA and run a finite action smoke test.", "M4", ("config",)
    ),
    "train_seen": CommandSpec(
        "Train the seen-domain policy with exact resume.",
        "M5",
        ("config", "resume_from"),
    ),
    "train_target": CommandSpec(
        "Adapt independently from the immutable seen checkpoint.",
        "M8",
        ("config", "task", "n_demos", "seed", "resume_from"),
    ),
    "eval_seen": CommandSpec(
        "Evaluate only the fixed seen probe suite.", "M6", ("config", "checkpoint")
    ),
    "eval_target": CommandSpec(
        "Run resumable fixed-seed target rollouts.",
        "M7",
        ("config", "checkpoint", "task"),
    ),
    "eval_language_control": CommandSpec(
        "Run paired correct/wrong instruction rollouts.",
        "M7",
        ("config", "checkpoint", "task"),
    ),
    "verify_checkpoint": CommandSpec(
        "Verify checkpoint completeness and checksums.", "M5", ("checkpoint",)
    ),
    "sync_artifacts": CommandSpec(
        "Checksummed, dry-run-first artifact synchronization.", "M5", ("execute",)
    ),
    "build_registry": CommandSpec("Rebuild registry rows from immutable manifests.", "M5"),
    "collect_results": CommandSpec("Validate and collect completed rollout records.", "M10"),
    "plot_cost_curve": CommandSpec("Build observed cost curves with uncertainty.", "M10"),
    "make_report_tables": CommandSpec("Build deterministic final report tables.", "M10"),
    "prune_artifacts": CommandSpec(
        "Inventory safe retention candidates; never delete by default.", "M5", ("execute",)
    ),
}


def build_stub_parser(command: str) -> argparse.ArgumentParser:
    spec = COMMANDS[command]
    parser = argparse.ArgumentParser(prog=f"scripts/{command}.py", description=spec.description)
    for argument in spec.arguments:
        if argument == "config":
            parser.add_argument("--config", type=Path, help="Tracked YAML config to validate.")
        elif argument == "metadata_only":
            parser.add_argument("--metadata-only", action="store_true")
        elif argument == "suite":
            parser.add_argument("--suite", choices=("libero_90", "libero_goal"))
        elif argument == "split":
            parser.add_argument(
                "--split",
                type=Path,
                default=Path("configs/splits/target_splits.json"),
            )
        elif argument == "task":
            parser.add_argument(
                "--task",
                choices=("drawer_middle", "bowl_stove", "wine_cabinet"),
            )
        elif argument == "episode_id":
            parser.add_argument("--episode-id", type=int)
        elif argument == "n_demos":
            parser.add_argument("--n-demos", type=int, choices=(5, 10, 25))
        elif argument == "seed":
            parser.add_argument("--seed", type=int, choices=(42, 123))
        elif argument == "save_video":
            parser.add_argument("--save-video", action="store_true")
        elif argument == "resume_from":
            parser.add_argument("--resume-from", type=Path)
        elif argument == "checkpoint":
            parser.add_argument("--checkpoint", type=str)
        elif argument == "execute":
            parser.add_argument(
                "--execute",
                action="store_true",
                help="Explicitly opt in; default remains dry-run.",
            )
    return parser


def refuse_until_milestone(command: str) -> int:
    """Exit before compute with a stable unavailable-milestone message."""

    spec = COMMANDS[command]
    print(
        f"{command} is intentionally unavailable until {spec.milestone}; "
        "no compute or external write was started.",
        file=sys.stderr,
    )
    return 2


def run_milestone_stub(command: str, argv: list[str] | None = None) -> int:
    """Parse the stable interface, then refuse execution until its milestone."""

    parser = build_stub_parser(command)
    args = parser.parse_args(argv)
    config_path = getattr(args, "config", None)
    if config_path is not None:
        try:
            load_config(config_path)
        except Exception as error:
            parser.error(f"invalid config: {error}")
    return refuse_until_milestone(command)
