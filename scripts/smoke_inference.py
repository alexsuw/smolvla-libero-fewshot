"""Load pinned SmolVLA, verify LIBERO features and trainable allowlist."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from vla_fewshot.config import TrainConfig, TrainableScope, load_config
from vla_fewshot.model.features import (
    LIBERO_INPUT_FEATURES,
    LIBERO_OUTPUT_FEATURES,
    assert_libero_policy_features,
)
from vla_fewshot.model.freezing import lerobot_finetune_flags
from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text

SmokeProfile = Literal["static", "full"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/smoke.yaml"))
    parser.add_argument(
        "--profile",
        choices=("static", "full"),
        default="full",
        help="static: feature/allowlist contracts only. full: load weights on CUDA.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--with-env",
        action="store_true",
        help="Step the LIBERO env with the converted action. Requires GPU extra.",
    )
    parser.add_argument("--task", default="put the bowl on the stove")
    return parser


def _write_static_report(
    *,
    output_dir: Path,
    scope: TrainableScope,
    repo_id: str,
    revision: str,
) -> dict:
    features = assert_libero_policy_features(
        input_features=LIBERO_INPUT_FEATURES,
        output_features=LIBERO_OUTPUT_FEATURES,
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "profile": "static",
        "acceptance_complete": False,
        "model_repo_id": repo_id,
        "model_revision": revision,
        "features": features,
        "trainable_scope": scope.model_dump(),
        "lerobot_flags": lerobot_finetune_flags(scope),
        "notes": (
            "Pinned smolvla_base hub features are SO100 (not LIBERO). "
            "The loader overlays LIBERO 2-camera/8D/7D features after load. "
            "Full CUDA inference is required before this gate is complete."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "smoke_inference.json", payload, overwrite=True)
    atomic_write_text(
        output_dir / "smoke_inference.md",
        "\n".join(
            [
                "# SmolVLA smoke (static)",
                "",
                f"- model: `{repo_id}@{revision}`",
                f"- action dim: `{features['output_features']['action']['shape']}`",
                f"- LeRobot flags: `{payload['lerobot_flags']}`",
                f"- acceptance_complete: `{payload['acceptance_complete']}`",
                "",
                payload["notes"],
                "",
            ]
        ),
        overwrite=True,
    )
    return payload


def _run_full(
    *,
    config: TrainConfig,
    output_dir: Path,
    task_text: str,
    with_env: bool,
) -> dict:
    from vla_fewshot.model.smolvla import load_pinned_smolvla, run_dummy_inference

    loaded = load_pinned_smolvla(
        repo_id=config.model.repo_id,
        revision=config.model.revision,
        scope=config.trainable_scope,
        output_dir=output_dir,
    )
    inference = run_dummy_inference(policy=loaded["policy"], task_text=task_text)
    env_step = None
    if with_env:
        from vla_fewshot.env.libero_env import LiberoRuntime, resolve_env_task_id

        runtime = LiberoRuntime(
            suite="libero_goal",
            task_id=resolve_env_task_id(
                suite="libero_goal",
                task_text=task_text,
                configured=None,
            ),
            seed=0,
        )
        try:
            runtime.reset(seed=0)
            _, _, _, _, info = runtime.step(inference["env_action"])
            env_step = {"accepted": True, "is_success": info.get("is_success")}
        finally:
            runtime.close()
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "profile": "full",
        "acceptance_complete": True,
        "model_repo_id": config.model.repo_id,
        "model_revision": config.model.revision,
        "device": loaded["device"],
        "feature_overlay": loaded["feature_overlay"],
        "trainable_scope": loaded["trainable_scope"],
        "inference": inference,
        "env_step": env_step,
    }
    atomic_write_json(output_dir / "smoke_inference.json", payload, overwrite=True)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    loaded = load_config(args.config)
    if not isinstance(loaded, TrainConfig):
        raise SystemExit(f"{args.config} is not a train config")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("artifacts/smoke_inference") / stamp
    try:
        if args.profile == "static":
            payload = _write_static_report(
                output_dir=output_dir,
                scope=loaded.trainable_scope,
                repo_id=loaded.model.repo_id,
                revision=loaded.model.revision,
            )
        else:
            payload = _run_full(
                config=loaded,
                output_dir=output_dir,
                task_text=args.task,
                with_env=args.with_env,
            )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in payload.items() if k != "policy"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
