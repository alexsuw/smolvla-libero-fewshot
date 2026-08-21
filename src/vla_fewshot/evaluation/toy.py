"""Deterministic toy env/policy used to prove the eval protocol without LIBERO."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from vla_fewshot.env.action_adapter import ACTION_DIM
from vla_fewshot.storage.checksums import sha256_json


def instruction_goal(instruction: str) -> tuple[float, float]:
    rng = random.Random(instruction)
    return rng.random(), rng.random()


@dataclass
class ToyEvalEnv:
    horizon: int
    seed: int | None = None
    instruction: str = ""
    x: float = 0.0
    y: float = 0.0
    t: int = 0

    def reset(self, *, seed: int, instruction: str) -> tuple[dict[str, Any], dict[str, Any]]:
        self.seed = seed
        self.instruction = instruction
        rng = random.Random(seed)
        self.x = rng.random()
        self.y = rng.random()
        self.t = 0
        return self._observation(), {}

    def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if len(action) != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}D action, got {len(action)}")
        self.x += float(action[0]) * 0.25
        self.y += float(action[1]) * 0.25
        self.t += 1
        goal = instruction_goal(self.instruction)
        dist = math.hypot(self.x - goal[0], self.y - goal[1])
        success = dist < 0.2
        terminated = success
        truncated = (not success) and self.t >= self.horizon
        info = {"is_success": success}
        return self._observation(), 0.0, terminated, truncated, info

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        return {
            "observation.state": [self.x, self.y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "instruction": self.instruction,
            "seed": self.seed,
        }

    def render_frame(self) -> list[list[list[int]]]:
        size = 8
        gx, gy = instruction_goal(self.instruction)
        px = min(size - 1, max(0, int(self.x * size)))
        py = min(size - 1, max(0, int(self.y * size)))
        cx = min(size - 1, max(0, int(gx * size)))
        cy = min(size - 1, max(0, int(gy * size)))
        frame = [[[20, 20, 20] for _ in range(size)] for _ in range(size)]
        frame[cy][cx] = [0, 180, 0]
        frame[py][px] = [220, 40, 40]
        return frame


class ToyEvalPolicy:
    """Moves toward the instruction goal so wrong language diverges."""

    def act(self, observation: dict[str, Any], *, chunk_size: int) -> list[list[float]]:
        state = observation["observation.state"]
        goal = instruction_goal(str(observation["instruction"]))
        dx = goal[0] - float(state[0])
        dy = goal[1] - float(state[1])
        step = [dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0]
        return [list(step) for _ in range(chunk_size)]


def fingerprint_observation(observation: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in observation.items()
        if key not in {"instruction"}
    }
    return "sha256:" + sha256_json(payload)
