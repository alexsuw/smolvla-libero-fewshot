"""75/25 target vs libero_90 replay mixing. Goal samples are forbidden."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from vla_fewshot.config import TrainConfig
from vla_fewshot.data.expected import SEEN_SUITE, TARGET_TASKS
from vla_fewshot.data.task_text import task_text_matches
from vla_fewshot.training.cursor import FrameCursor


def _train_error(message: str) -> None:
    from vla_fewshot.training.trainer import TrainError as _TrainError

    raise _TrainError(message)


def assert_replay_disabled(config: TrainConfig) -> None:
    if config.method == "replay_lora" and config.replay is not None and config.replay.enabled:
        return
    if config.replay is not None and config.replay.enabled:
        raise RuntimeError(
            "replay mixing is not part of the M5 smoke or primary seen path"
        )


def assert_replay_lora_train_config(config: TrainConfig) -> None:
    if config.stage != "target" or config.method != "replay_lora":
        _train_error(
            "Replay-LoRA path requires stage=target method=replay_lora. "
            "no GPU training was started."
        )
    if config.peft is None:
        _train_error("Replay-LoRA requires peft. no GPU training was started.")
    if config.replay is None or not config.replay.enabled:
        _train_error("Replay-LoRA requires replay.enabled. no GPU training was started.")
    if config.replay.seen_suite != SEEN_SUITE:
        _train_error("Replay-LoRA replay pool must be libero_90. no GPU training was started.")
    if abs(config.replay.target_fraction - 0.75) > 1e-12 or abs(config.replay.seen_fraction - 0.25) > 1e-12:
        _train_error("Replay-LoRA fractions must stay 0.75/0.25. no GPU training was started.")
    if config.dataset.suite != "libero_goal":
        _train_error("Replay-LoRA target episodes stay libero_goal selected")
    if not config.training.sample_with_replacement:
        _train_error("Replay-LoRA requires sample_with_replacement: true")
    if config.trainable_scope.train_action_expert:
        _train_error(
            "Replay-LoRA keeps Action Expert frozen except adapters. "
            "no GPU training was started."
        )
    if not config.trainable_scope.freeze_vlm_backbone or not config.trainable_scope.freeze_vision_encoder:
        _train_error("Replay-LoRA keeps VLM/vision frozen. no GPU training was started.")
    from vla_fewshot.calibration import assert_train_matches_calibration, load_calibration

    assert_train_matches_calibration(config, load_calibration())


def assert_replay_pool(*, suite: str, task_texts: list[str]) -> None:
    if suite != SEEN_SUITE:
        _train_error(
            f"replay pool suite must be {SEEN_SUITE}, not {suite}. "
            "no GPU training was started."
        )
    for spec in TARGET_TASKS.values():
        text = str(spec["task_text"])
        if any(task_text_matches(item, text) for item in task_texts):
            _train_error(
                f"libero_goal task text leaked into replay pool: {text!r}. "
                "no GPU training was started."
            )


def assert_replay_sample_not_goal(sample: Mapping[str, Any]) -> None:
    raw = sample.get("task", sample.get("task_index"))
    if raw is None:
        return
    texts = raw if isinstance(raw, (list, tuple)) else [raw]
    for item in texts:
        if not isinstance(item, str):
            continue
        for spec in TARGET_TASKS.values():
            if task_text_matches(item, str(spec["task_text"])):
                _train_error(
                    "libero_goal sample appeared in replay. no GPU training was started."
                )


def split_batch_counts(
    batch_size: int,
    *,
    target_fraction: float,
    remainder: float,
) -> tuple[int, int, float]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    exact = batch_size * target_fraction + remainder
    n_target = int(exact)
    if n_target > batch_size:
        n_target = batch_size
    if n_target < 0:
        n_target = 0
    n_replay = batch_size - n_target
    return n_target, n_replay, exact - n_target


@dataclass
class MixDraw:
    sources: list[str]
    target_indices: list[int]
    replay_indices: list[int]

    @property
    def n_target(self) -> int:
        return len(self.target_indices)

    @property
    def n_replay(self) -> int:
        return len(self.replay_indices)

    @property
    def target_fraction(self) -> float:
        total = self.n_target + self.n_replay
        return self.n_target / total if total else 0.0

    @property
    def seen_fraction(self) -> float:
        total = self.n_target + self.n_replay
        return self.n_replay / total if total else 0.0


def gather_mixed_samples(
    draw: MixDraw,
    *,
    target_dataset: Any,
    replay_dataset: Any,
) -> list[Any]:
    t_iter = iter(draw.target_indices)
    r_iter = iter(draw.replay_indices)
    samples: list[Any] = []
    for source in draw.sources:
        if source == "target":
            samples.append(target_dataset[next(t_iter)])
            continue
        sample = replay_dataset[next(r_iter)]
        assert_replay_sample_not_goal(sample)
        samples.append(sample)
    return samples


class ReplayMixer:
    """Deterministic 75/25 mix. Target N is unchanged; replay has its own cursor."""

    def __init__(
        self,
        *,
        n_target: int,
        n_replay: int,
        target_fraction: float,
        seen_fraction: float,
        seed: int,
        with_replacement: bool,
    ) -> None:
        if abs(target_fraction + seen_fraction - 1.0) > 1e-9:
            _train_error("replay fractions must sum to 1")
        if n_target < 1 or n_replay < 1:
            _train_error("Replay-LoRA needs non-empty target and libero_90 pools")
        self.target_fraction = target_fraction
        self.seen_fraction = seen_fraction
        self.seed = seed
        self.target = FrameCursor.create(n_target, seed=seed, with_replacement=with_replacement)
        self.replay = FrameCursor.create(n_replay, seed=seed + 1, with_replacement=True)
        self.rng = random.Random(seed + 2)
        self.remainder = 0.0
        self.cum_target = 0
        self.cum_replay = 0

    def next_draw(self, batch_size: int) -> MixDraw:
        n_target, n_replay, self.remainder = split_batch_counts(
            batch_size, target_fraction=self.target_fraction, remainder=self.remainder
        )
        labels = ["target"] * n_target + ["replay"] * n_replay
        self.rng.shuffle(labels)
        target_indices = self.target.next_indices(n_target) if n_target else []
        replay_indices = self.replay.next_indices(n_replay) if n_replay else []
        t_iter = iter(target_indices)
        r_iter = iter(replay_indices)
        ordered_target: list[int] = []
        ordered_replay: list[int] = []
        for label in labels:
            if label == "target":
                ordered_target.append(next(t_iter))
            else:
                ordered_replay.append(next(r_iter))
        self.cum_target += n_target
        self.cum_replay += n_replay
        return MixDraw(
            sources=labels,
            target_indices=ordered_target,
            replay_indices=ordered_replay,
        )

    def cumulative_fractions(self) -> tuple[float, float]:
        total = self.cum_target + self.cum_replay
        if not total:
            return 0.0, 0.0
        return self.cum_target / total, self.cum_replay / total

    def state_dict(self) -> dict[str, object]:
        version, numbers, gauss = self.rng.getstate()
        return {
            "kind": "replay_mixer",
            "target_fraction": self.target_fraction,
            "seen_fraction": self.seen_fraction,
            "seed": self.seed,
            "remainder": self.remainder,
            "cum_target": self.cum_target,
            "cum_replay": self.cum_replay,
            "target": self.target.state_dict(),
            "replay": self.replay.state_dict(),
            "rng_version": version,
            "rng_numbers": list(numbers),
            "rng_gauss": gauss,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("kind") != "replay_mixer":
            raise ValueError("sampler is not a replay mixer")
        if abs(float(state["target_fraction"]) - self.target_fraction) > 1e-12:
            raise ValueError("replay mixer target_fraction mismatch")
        self.remainder = float(state["remainder"])
        self.cum_target = int(state["cum_target"])
        self.cum_replay = int(state["cum_replay"])
        self.target.load_state_dict(state["target"])  # type: ignore[arg-type]
        self.replay.load_state_dict(state["replay"])  # type: ignore[arg-type]
        self.rng.setstate(
            (
                int(state["rng_version"]),
                tuple(state["rng_numbers"]),  # type: ignore[arg-type]
                state["rng_gauss"],
            )
        )
