"""Deterministic frame-index cursor for resume-safe batching."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class FrameCursor:
    """Shuffled cyclic indices, or seeded with-replacement draws."""

    n_samples: int
    seed: int
    with_replacement: bool
    order: list[int]
    cursor: int
    epoch: int
    rng: random.Random

    @classmethod
    def create(cls, n_samples: int, *, seed: int, with_replacement: bool) -> "FrameCursor":
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        rng = random.Random(seed)
        order = list(range(n_samples))
        rng.shuffle(order)
        return cls(
            n_samples=n_samples,
            seed=seed,
            with_replacement=with_replacement,
            order=order,
            cursor=0,
            epoch=0,
            rng=rng,
        )

    def next_indices(self, batch_size: int) -> list[int]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.with_replacement:
            return [self.rng.randrange(self.n_samples) for _ in range(batch_size)]
        indices: list[int] = []
        for _ in range(batch_size):
            if self.cursor >= len(self.order):
                self.epoch += 1
                self.rng.seed(self.seed + self.epoch)
                self.order = list(range(self.n_samples))
                self.rng.shuffle(self.order)
                self.cursor = 0
            indices.append(self.order[self.cursor])
            self.cursor += 1
        return indices

    def state_dict(self) -> dict[str, object]:
        version, numbers, gauss = self.rng.getstate()
        return {
            "n_samples": self.n_samples,
            "seed": self.seed,
            "with_replacement": self.with_replacement,
            "order": list(self.order),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "rng_version": version,
            "rng_numbers": list(numbers),
            "rng_gauss": gauss,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["n_samples"]) != self.n_samples:
            raise ValueError("frame cursor n_samples mismatch")
        if bool(state["with_replacement"]) != self.with_replacement:
            raise ValueError("frame cursor replacement mismatch")
        self.order = list(state["order"])  # type: ignore[arg-type]
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        self.rng.setstate(
            (
                int(state["rng_version"]),
                tuple(state["rng_numbers"]),  # type: ignore[arg-type]
                state["rng_gauss"],
            )
        )
