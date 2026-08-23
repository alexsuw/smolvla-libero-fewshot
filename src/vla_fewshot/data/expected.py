"""Pinned-revision counts and target texts from PROJECT_SPEC.md."""

from __future__ import annotations

from typing import Final

SEEN_SUITE: Final = "libero_90"
TARGET_SUITE: Final = "libero_goal"
DEMO_BUDGETS: Final = (5, 10, 25)
LOW_N_BUDGETS: Final = (1, 2)
PREFIX_BUDGETS: Final = LOW_N_BUDGETS + DEMO_BUDGETS

EXPECTED_SUITE_COUNTS: Final[dict[str, dict[str, int]]] = {
    SEEN_SUITE: {
        "episodes": 3921,
        "frames": 569249,
        "unique_task_texts": 73,
        "fps": 20,
    },
    TARGET_SUITE: {
        "episodes": 428,
        "frames": 52042,
        "unique_task_texts": 10,
        "fps": 20,
    },
}

TARGET_TASKS: Final[dict[str, dict[str, object]]] = {
    "drawer_middle": {
        "task_text": "open the middle drawer of the cabinet",
        "task_index": 9,
        "available_count": 43,
    },
    "bowl_stove": {
        "task_text": "put the bowl on the stove",
        "task_index": 7,
        "available_count": 48,
    },
    "wine_cabinet": {
        "task_text": "put the wine bottle on top of the cabinet",
        "task_index": 4,
        "available_count": 47,
    },
}

EXPECTED_FEATURES: Final[dict[str, dict[str, object]]] = {
    "action": {"dtype": "float32", "shape": [7]},
    "observation.state": {"dtype": "float32", "shape": [8]},
    "observation.images.image": {"shape": [256, 256, 3]},
    "observation.images.wrist_image": {"shape": [256, 256, 3]},
}
