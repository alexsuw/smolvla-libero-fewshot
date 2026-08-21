"""Shared reporting constants. Cost-curve x ticks stay exact."""

from __future__ import annotations

COST_CURVE_N = (0, 5, 10, 25)
TARGET_TASK_SLUGS = ("drawer_middle", "bowl_stove", "wine_cabinet")
TARGET_METHODS = ("baseline", "lora", "replay_lora")
TRAIN_SEEDS = (42, 123)
REPORT_PROTOCOL_IDS = frozenset({"final_v1", "language_control_v1", "seen_probe_v1"})
EXCLUDED_PROTOCOL_PREFIXES = ("static_",)
EXCLUDED_PROTOCOL_IDS = frozenset({"dev_soft_reset"})

LONG_COLUMNS = (
    "eval_run_id",
    "method",
    "task_slug",
    "n_demos",
    "train_seed",
    "eval_seed",
    "instruction_condition",
    "protocol_id",
    "success",
    "checkpoint_sha256",
    "suite",
)
