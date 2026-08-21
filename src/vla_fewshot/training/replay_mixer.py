"""Replay mixing is disabled for M5 smoke / seen-pretrain primary path."""

from __future__ import annotations

from vla_fewshot.config import TrainConfig


def assert_replay_disabled(config: TrainConfig) -> None:
    if config.replay is not None and config.replay.enabled:
        raise RuntimeError(
            "replay mixing is not part of the M5 smoke or primary seen path"
        )
