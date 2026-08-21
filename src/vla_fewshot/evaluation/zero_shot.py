"""Zero-shot final eval: 3 target tasks × ≥20 from the frozen seen checkpoint."""

from __future__ import annotations

from pathlib import Path

from vla_fewshot.calibration import load_selected_checkpoint
from vla_fewshot.config import EvalConfig
from vla_fewshot.data.expected import TARGET_TASKS
from vla_fewshot.evaluation.protocol import ProtocolError

ZERO_SHOT_SLUGS = tuple(TARGET_TASKS)
MIN_ROLLOUTS = 20


def assert_zero_shot_config(config: EvalConfig, *, profile: str) -> None:
    if config.stage != "zero_shot":
        raise ProtocolError("zero-shot evaluation requires configs/eval/zero_shot.yaml")
    if not config.protocol.hard_reset:
        raise ProtocolError("zero-shot requires hard_reset: true")
    if profile == "full":
        if config.protocol.protocol_id != "final_v1":
            raise ProtocolError("zero-shot full eval must use protocol_id=final_v1")
        if config.protocol.rollouts_per_cell < MIN_ROLLOUTS:
            raise ProtocolError(
                f"zero-shot requires ≥{MIN_ROLLOUTS} rollouts per task"
            )


def assert_zero_shot_cell(
    *,
    n_demos: int | None,
    train_seed: int | None,
    episode_ids: list[int],
) -> None:
    if n_demos not in (None, 0):
        raise ProtocolError("zero-shot uses 0 target demonstrations")
    if train_seed is not None:
        raise ProtocolError("zero-shot has no adaptation train seed")
    if episode_ids:
        raise ProtocolError("zero-shot training episode list must be empty")


def resolve_frozen_eval_checkpoint(checkpoint: Path | None) -> tuple[Path, str]:
    selected = load_selected_checkpoint()
    if selected.status != "frozen" or not selected.sha256 or selected.uri is None:
        raise RuntimeError(
            "zero-shot waits until configs/selected_seen_checkpoint.yaml is "
            "frozen from seen probes. no GPU evaluation was started."
        )
    origin = Path(checkpoint) if checkpoint is not None else Path(selected.uri)
    return origin, selected.sha256


def assert_frozen_checkpoint_hash(path: Path, expected_sha256: str) -> str:
    from vla_fewshot.evaluation.runner import checkpoint_sha256

    digest = checkpoint_sha256(path)
    if digest != expected_sha256:
        raise RuntimeError(
            f"checkpoint hash {digest} != frozen seen {expected_sha256}. "
            "no GPU evaluation was started."
        )
    return digest


def zero_shot_commands(
    *,
    config: Path = Path("configs/eval/zero_shot.yaml"),
) -> list[list[str]]:
    return [
        [
            "python",
            "scripts/eval_zero_shot.py",
            "--config",
            str(config),
            "--task",
            task,
            "--profile",
            "full",
        ]
        for task in ZERO_SHOT_SLUGS
    ]
