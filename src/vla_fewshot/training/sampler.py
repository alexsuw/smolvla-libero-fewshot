"""Deterministic cyclic sampler with serializable cursor state."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Sample:
    index: int
    x: float
    y: float


class DeterministicSampler:
    """Fixed synthetic pairs. Real dataset iteration is a later GPU milestone."""

    def __init__(self, *, seed: int, n_samples: int = 16) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        self.n_samples = n_samples
        rng = random.Random(seed)
        pairs = []
        for index in range(n_samples):
            x = float(index + 1) / float(n_samples)
            y = 0.3 * x + 0.1
            pairs.append((index, x, y))
        order = list(range(n_samples))
        rng.shuffle(order)
        self.indices = order
        self.pairs = pairs
        self.cursor = 0

    def next_sample(self) -> Sample:
        index = self.indices[self.cursor % self.n_samples]
        self.cursor += 1
        raw_index, x, y = self.pairs[index]
        return Sample(index=raw_index, x=x, y=y)

    def order_hash_payload(self) -> list[int]:
        return list(self.indices)

    def state_dict(self) -> dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "indices": list(self.indices),
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["n_samples"]) != self.n_samples:
            raise ValueError("sampler n_samples mismatch")
        indices = list(state["indices"])  # type: ignore[arg-type]
        if indices != self.indices:
            raise ValueError("sampler index order mismatch")
        self.cursor = int(state["cursor"])
