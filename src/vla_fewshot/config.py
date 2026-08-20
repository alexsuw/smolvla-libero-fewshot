"""Strict, project-owned configuration contracts.

M0 intentionally validates intent without importing CUDA, LeRobot, or LIBERO.
Pinned upstream adapters are added only after their exact APIs are inspected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that fails on misspelled or unrecognised keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RevisionRef(StrictModel):
    repo_id: str = Field(min_length=3)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class SourcePins(StrictModel):
    lerobot_git: str = Field(pattern=r"^[0-9a-f]{40}$")
    libero_runtime_package: str
    libero_upstream_reference: str = Field(pattern=r"^[0-9a-f]{40}$")


class RuntimePins(StrictModel):
    torch: str
    torchvision: str
    cuda_wheel_variant: str
    mujoco: str
    transformers: str
    peft: str
    accelerate: str
    numpy: str
    ffmpeg: str


class DatasetPins(RevisionRef):
    suite_seen: Literal["libero_90"]
    suite_target: Literal["libero_goal"]


class RevisionsConfig(StrictModel):
    kind: Literal["revisions"]
    schema_version: Literal[1]
    status: Literal["provisional_m0", "validated_m1"]
    snapshot_date: str
    python: str
    dataset: DatasetPins
    model: RevisionRef
    source: SourcePins
    runtime: RuntimePins


class StorageConfig(StrictModel):
    kind: Literal["storage"]
    schema_version: Literal[1]
    data_root_env: Literal["VLA_DATA_ROOT"]
    datasets_dir_env: Literal["VLA_DATASETS_DIR"]
    runs_dir_env: Literal["VLA_RUNS_DIR"]
    checkpoints_dir_env: Literal["VLA_CHECKPOINTS_DIR"]
    cache_dir_env: Literal["VLA_CACHE_DIR"]
    scratch_dir_env: Literal["VLA_SCRATCH_DIR"]
    object_uri_env: Literal["VLA_OBJECT_URI"]
    overwrite: Literal[False]
    sync_dry_run_by_default: Literal[True]


class RuntimeConfig(StrictModel):
    os: Literal["linux"]
    python: Literal["3.12"]
    mujoco_gl: Literal["egl"]
    tokenizers_parallelism: Literal[False]


class HardwareConfig(StrictModel):
    device: Literal["auto", "cuda"]
    mixed_precision: Literal["auto", "bf16", "fp16", "fp32"]
    allow_tf32: bool
    physical_batch_size: int | Literal["auto_fit"]


class StoragePolicy(StrictModel):
    durable: bool
    reserve_gb: int = Field(ge=1)
    require_verified_backup: bool


class PreemptionConfig(StrictModel):
    enabled: bool
    signals: list[Literal["SIGTERM", "SIGINT"]]
    emergency_checkpoint: bool
    flush_logs: bool
    bounded_sync_seconds: int = Field(ge=0)


class PlatformConfig(StrictModel):
    kind: Literal["platform"]
    schema_version: Literal[1]
    name: Literal["colab", "gpu_vm"]
    runtime: RuntimeConfig
    hardware: HardwareConfig
    storage: StoragePolicy
    preemption: PreemptionConfig


class TargetTask(StrictModel):
    task_text: str = Field(min_length=1)
    task_index: int = Field(ge=0)
    available_count: int = Field(ge=25)


class DataConfig(StrictModel):
    kind: Literal["data"]
    schema_version: Literal[1]
    dataset: DatasetPins
    exact_text_matching: Literal[True]
    targets: dict[str, TargetTask]

    @model_validator(mode="after")
    def require_three_targets(self) -> "DataConfig":
        required = {"drawer_middle", "bowl_stove", "wine_cabinet"}
        if set(self.targets) != required:
            raise ValueError(f"targets must be exactly {sorted(required)}")
        return self


class ModelConfig(RevisionRef):
    pass


class TrainDatasetConfig(RevisionRef):
    suite: Literal["libero_90", "libero_goal"]
    episodes: Literal["all", "selected"]
    max_tasks: int | None = Field(default=None, ge=1)
    max_episodes: int | None = Field(default=None, ge=1)


class TrainableScope(StrictModel):
    freeze_vision_encoder: bool
    freeze_vlm_backbone: bool
    train_action_expert: bool
    train_state_projection: bool
    train_action_projections: bool
    strict_allowlist: Literal[True]


class OptimizerConfig(StrictModel):
    name: Literal["adamw"]
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    betas: tuple[float, float]
    eps: float = Field(gt=0)


class SchedulerConfig(StrictModel):
    name: Literal["cosine"]
    warmup_steps: int = Field(ge=0)
    min_lr: float = Field(ge=0)


class TrainingConfig(StrictModel):
    max_steps: int = Field(ge=1)
    epochs: int | None = Field(default=None, ge=1)
    physical_batch_size: int | Literal["auto_fit"]
    effective_batch_size: int = Field(ge=1)
    gradient_accumulation: int | Literal["auto"]
    max_grad_norm: float = Field(gt=0)
    mixed_precision: Literal["auto", "bf16", "fp16", "fp32"]
    seed: int
    num_workers: int = Field(ge=0)
    sample_with_replacement: bool


class CheckpointConfig(StrictModel):
    every_steps: int | None = Field(default=None, ge=1)
    save_steps: list[int] = Field(default_factory=list)
    milestones: list[int] = Field(default_factory=list)
    atomic: Literal[True]
    verify_fresh_load: Literal[True]


class TrackingConfig(StrictModel):
    wandb_enabled: Literal[False]
    tensorboard_enabled: Literal[True]
    csv_enabled: Literal[True]
    jsonl_events_enabled: Literal[True]


class PeftConfig(StrictModel):
    method_type: Literal["LORA"]
    r: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    lora_dropout: float = Field(ge=0, lt=1)


class ReplayConfig(StrictModel):
    enabled: bool
    target_fraction: float = Field(gt=0, le=1)
    seen_fraction: float = Field(ge=0, lt=1)
    seen_suite: Literal["libero_90"] | None = None

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "ReplayConfig":
        if abs(self.target_fraction + self.seen_fraction - 1.0) > 1e-9:
            raise ValueError("replay fractions must sum to 1")
        return self


class TrainConfig(StrictModel):
    kind: Literal["train"]
    schema_version: Literal[1]
    stage: Literal["smoke", "seen", "target"]
    method: Literal["smoke", "expert", "lora", "baseline", "replay_lora"]
    model: ModelConfig
    dataset: TrainDatasetConfig
    trainable_scope: TrainableScope
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    tracking: TrackingConfig
    peft: PeftConfig | None = None
    replay: ReplayConfig | None = None


class EvaluationProtocol(StrictModel):
    protocol_id: str = Field(min_length=1)
    hard_reset: bool
    rollouts_per_cell: int = Field(ge=1)
    seeds_file: str
    max_horizon: int = Field(ge=1)
    action_chunk_horizon: int = Field(ge=1)
    deterministic: bool
    save_every_failure_video: Literal[True]
    save_first_success_video: Literal[True]
    save_every_trace: Literal[True]


class EvalConfig(StrictModel):
    kind: Literal["eval"]
    schema_version: Literal[1]
    stage: Literal["seen_probe", "zero_shot", "language_control", "final"]
    dataset: DatasetPins
    protocol: EvaluationProtocol
    tracking: TrackingConfig
    wrong_instruction_map: dict[str, str] | None = None


ConfigModel = (
    RevisionsConfig
    | StorageConfig
    | PlatformConfig
    | DataConfig
    | TrainConfig
    | EvalConfig
)

MODEL_BY_KIND: dict[str, type[StrictModel]] = {
    "revisions": RevisionsConfig,
    "storage": StorageConfig,
    "platform": PlatformConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}

FORBIDDEN_PATH_PREFIXES = ("/content", "/mnt/vla")


def _reject_hard_coded_paths(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_hard_coded_paths(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_hard_coded_paths(child, f"{location}[{index}]")
    elif isinstance(value, str) and value.startswith(FORBIDDEN_PATH_PREFIXES):
        raise ValueError(f"hard-coded platform path at {location}: {value}")


def load_config(path: str | Path) -> ConfigModel:
    """Load a tracked YAML config and reject unknown keys or unsafe paths."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    _reject_hard_coded_paths(raw)
    kind = raw.get("kind")
    model = MODEL_BY_KIND.get(kind)
    if model is None:
        raise ValueError(
            f"{config_path} has unknown kind {kind!r}; "
            f"expected one of {sorted(MODEL_BY_KIND)}"
        )
    return model.model_validate(raw)


def discover_configs(root: str | Path) -> list[Path]:
    """Return tracked YAML configs in deterministic order."""

    return sorted(Path(root).rglob("*.yaml"))
