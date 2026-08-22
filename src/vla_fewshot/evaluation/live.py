"""Live SmolVLA/LIBERO rollouts. Imported only after the CUDA runtime gate."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.config import TrainConfig, TrainableScope
from vla_fewshot.data.layout import dataset_revision_root
from vla_fewshot.data.metadata import load_suite_metadata
from vla_fewshot.env.action_adapter import dataset_action_to_env
from vla_fewshot.env.libero_env import LiberoRuntime, resolve_env_task_id
from vla_fewshot.env.replay import load_replay_gate
from vla_fewshot.evaluation.protocol import PlannedRollout
from vla_fewshot.evaluation.toy import fingerprint_observation
from vla_fewshot.model.features import (
    POLICY_MAIN_IMAGE,
    POLICY_STATE,
    POLICY_TASK,
    POLICY_WRIST_IMAGE,
)
from vla_fewshot.model.smolvla import load_pinned_smolvla
from vla_fewshot.storage.layout import CHECKPOINT_WEIGHTS_PT_NAME
from vla_fewshot.training.checkpoint import is_complete_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GATE_ENV_TASK_IDS: dict[tuple[str, str], int] | None = None


def _replay_gate_env_task_id(suite: str, task_text: str) -> int | None:
    """Disambiguate duplicate LIBERO languages using the tracked replay gate."""

    global _GATE_ENV_TASK_IDS
    if _GATE_ENV_TASK_IDS is None:
        gate = load_replay_gate(_REPO_ROOT / "configs" / "splits" / "replay_gate.json")
        _GATE_ENV_TASK_IDS = {
            (item.suite, item.task_text): item.env_task_id
            for item in gate.episodes
            if item.env_task_id is not None
        }
    return _GATE_ENV_TASK_IDS.get((suite, task_text))


def _as_state_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _as_nchw_float(image: Any, device: Any) -> Any:
    import torch

    tensor = torch.as_tensor(image)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[-1] in {1, 3} and tensor.shape[0] not in {1, 3}:
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.unsqueeze(0)


def observation_to_batch(observation: dict[str, Any], *, task: str, device: Any) -> dict[str, Any]:
    import torch

    state = _as_state_list(observation[POLICY_STATE])
    return {
        POLICY_MAIN_IMAGE: _as_nchw_float(observation[POLICY_MAIN_IMAGE], device),
        POLICY_WRIST_IMAGE: _as_nchw_float(observation[POLICY_WRIST_IMAGE], device),
        POLICY_STATE: torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0),
        POLICY_TASK: [task],
    }


def _make_preprocessor(policy: Any, stats: dict[str, Any], device: str) -> Any:
    from lerobot.policies.factory import make_pre_post_processors

    try:
        preprocessor, _post = make_pre_post_processors(
            policy.config,
            pretrained_path=None,
            dataset_stats=stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    except TypeError:
        preprocessor, _post = make_pre_post_processors(
            policy.config,
            pretrained_path=None,
            dataset_stats=stats,
        )
    return preprocessor


def load_eval_policy(
    *,
    checkpoint: Path,
    repo_id: str,
    revision: str,
    scope: TrainableScope,
    stats: dict[str, Any],
    action_chunk_horizon: int,
    train: TrainConfig | None = None,
) -> dict[str, Any]:
    loaded = load_pinned_smolvla(
        repo_id=repo_id,
        revision=revision,
        scope=scope,
        device="cuda",
    )
    policy = loaded["policy"]
    peft = (
        train.peft
        if train is not None and train.method in {"lora", "replay_lora"}
        else None
    )
    if peft is not None:
        from vla_fewshot.model.peft import load_lora_policy_weights, wrap_policy_lora

        assert train is not None
        policy = wrap_policy_lora(policy, train)
        loaded["policy"] = policy
        if checkpoint.is_dir() and is_complete_checkpoint(checkpoint):
            load_lora_policy_weights(policy, checkpoint, peft=peft)
        elif checkpoint.is_file():
            raise RuntimeError(f"full eval expects a checkpoint directory, got file {checkpoint}")
    elif checkpoint.is_dir() and is_complete_checkpoint(checkpoint):
        import torch

        weights = torch.load(
            checkpoint / CHECKPOINT_WEIGHTS_PT_NAME,
            map_location=loaded["device"],
            weights_only=True,
        )
        policy.load_state_dict(weights)
    elif checkpoint.is_file():
        raise RuntimeError(f"full eval expects a checkpoint directory, got file {checkpoint}")
    policy.config.n_action_steps = min(int(action_chunk_horizon), int(policy.config.chunk_size))
    policy.eval()
    preprocessor = _make_preprocessor(policy, stats, loaded["device"])
    return {**loaded, "preprocessor": preprocessor}


def suite_stats(*, datasets_dir: Path, repo_id: str, revision: str, suite: str) -> dict[str, Any]:
    meta = load_suite_metadata(dataset_revision_root(datasets_dir, repo_id, revision), suite)
    if not meta.stats:
        raise RuntimeError(f"missing stats.json for {suite}; identity stats are forbidden for eval")
    return meta.stats


class LiveRolloutAdapter:
    """One SmolVLA + cached LIBERO env per task_id."""

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        device: Any,
        hard_reset: bool = True,
    ) -> None:
        if not hard_reset:
            raise ValueError("hard_reset must stay true")
        self.policy = policy
        self.preprocessor = preprocessor
        self.device = device
        self._envs: dict[tuple[str, int], LiberoRuntime] = {}

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()

    def _env(self, suite: str, task_id: int) -> LiberoRuntime:
        key = (suite, task_id)
        if key not in self._envs:
            self._envs[key] = LiberoRuntime(suite=suite, task_id=task_id, seed=0, hard_reset=True)
        return self._envs[key]

    def __call__(
        self,
        *,
        config: Any,
        spec: PlannedRollout,
        checkpoint_uri: str,
        checkpoint_sha256: str,
        eval_run_id: str,
        method: str,
        stage: str,
        episode_ids: list[int],
        git_commit: str | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any]]:
        import torch

        task_id = resolve_env_task_id(
            suite=spec.suite,
            task_text=spec.task_text,
            configured=_replay_gate_env_task_id(spec.suite, spec.task_text),
        )
        env = self._env(spec.suite, task_id)
        observation, _info = env.reset(seed=spec.eval_seed)
        fingerprint = fingerprint_observation(observation)
        self.policy.reset()
        traces: list[dict[str, Any]] = []
        frames: list[Any] = [env.extract_main_hwc(observation)]
        terminated = False
        truncated = False
        success = False
        started = time.perf_counter()
        steps = 0
        chunk_size = config.protocol.action_chunk_horizon
        while steps < config.protocol.max_horizon:
            batch = observation_to_batch(
                observation, task=spec.instruction_text_used, device=self.device
            )
            processed = self.preprocessor(batch)
            chunk: list[list[float]] = []
            with torch.inference_mode():
                for _ in range(chunk_size):
                    action = self.policy.select_action(processed)
                    tensor = action if torch.is_tensor(action) else torch.as_tensor(action)
                    if tensor.ndim == 3:
                        tensor = tensor[:, 0]
                    if tensor.ndim == 2:
                        tensor = tensor[0]
                    values = [float(item) for item in tensor.detach().cpu().flatten().tolist()]
                    chunk.append(values[:7])
            stop = False
            for dataset_action in chunk:
                env_action = dataset_action_to_env(dataset_action, binary=True)
                observation, _reward, terminated, truncated, info = env.step(list(env_action))
                success = bool(info.get("is_success"))
                traces.append(
                    {
                        "step": steps,
                        "action": list(env_action),
                        "dataset_action": list(dataset_action),
                        "state": _as_state_list(observation["observation.state"]),
                        "is_success": success,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                frames.append(env.extract_main_hwc(observation))
                steps += 1
                if terminated or truncated or success or steps >= config.protocol.max_horizon:
                    stop = True
                    break
            if stop:
                break
        record = {
            "schema_version": 1,
            "eval_run_id": eval_run_id,
            "train_run_id": None,
            "stage": stage,
            "method": method,
            "task_slug": spec.task_slug,
            "task_text": spec.task_text,
            "suite": spec.suite,
            "task_index": spec.task_index,
            "n_demos": 0 if spec.n_demos is None else spec.n_demos,
            "train_seed": spec.train_seed,
            "eval_seed": spec.eval_seed,
            "rollout_index": spec.rollout_index,
            "protocol_id": spec.protocol_id,
            "instruction_condition": spec.instruction_condition,
            "instruction_text_used": spec.instruction_text_used,
            "checkpoint_uri": checkpoint_uri,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_revision": config.dataset.revision,
            "training_episode_ids": episode_ids,
            "success": int(success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_length": steps,
            "wall_time_seconds": time.perf_counter() - started,
            "initial_state_fingerprint": fingerprint,
            "video_uri": None,
            "trace_uri": None,
            "failure_category": None if success else "unknown",
            "notes": "live SmolVLA/LIBERO rollout",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
        }
        return record, traces, frames
