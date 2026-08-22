"""Shared reporting constants. Cost-curve x ticks stay exact."""

from __future__ import annotations

COST_CURVE_N = (0, 5, 10, 25)
TARGET_TASK_SLUGS = ("drawer_middle", "bowl_stove", "wine_cabinet")
TARGET_METHODS = ("baseline", "lora", "replay_lora")
TRAIN_SEEDS = (42, 123)
LANGUAGE_CONTROL_PROTOCOL_IDS = frozenset(
    {"final_language_control_v1", "language_control_v1"}
)
ZERO_SHOT_PROTOCOL_ID = "zero_shot_v2_seen_stats"
ZERO_SHOT_PROTOCOL_IDS = frozenset({ZERO_SHOT_PROTOCOL_ID})
REPORT_PROTOCOL_IDS = (
    frozenset({"final_v1", "seen_probe_v1"})
    | ZERO_SHOT_PROTOCOL_IDS
    | LANGUAGE_CONTROL_PROTOCOL_IDS
)


def is_language_control_protocol(protocol_id: str) -> bool:
    return protocol_id in LANGUAGE_CONTROL_PROTOCOL_IDS
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
PROBE_EXPORT_COLUMNS = LONG_COLUMNS + (
    "step",
    "in_selection_pool",
    "video_uri",
    "trace_uri",
    "episode_length",
    "wall_time_seconds",
    "task_text",
    "instruction_text_used",
    "failure_category",
    "created_at_utc",
    "checkpoint_uri",
    "rollout_index",
    "terminated",
    "truncated",
)
