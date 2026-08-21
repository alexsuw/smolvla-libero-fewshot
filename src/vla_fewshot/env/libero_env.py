"""Pinned-LeRobot LIBERO runtime used by doctor, parity, and expert replay."""

from __future__ import annotations

from typing import Any

from vla_fewshot.env.observation_adapter import (
    apply_canonical_image_keys,
    flatten_libero_robot_state,
)


def _to_torch_tree(value: Any, torch: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_torch_tree(child, torch) for key, child in value.items()}
    return torch.as_tensor(value)


def require_libero_runtime() -> None:
    """Fail closed unless the pinned Linux LIBERO extra can be imported."""

    import platform

    if platform.system() != "Linux":
        raise RuntimeError(
            f"LIBERO runtime requires Linux EGL; current host is {platform.system()}"
        )
    try:
        import gymnasium  # noqa: F401
        import lerobot.envs.libero  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "LIBERO runtime requires `uv sync --frozen --extra gpu` on Linux"
        ) from error


def resolve_env_task_id(*, suite: str, task_text: str, configured: int | None) -> int:
    """Map exact instruction text to a LIBERO suite task_id.

    Dataset `task_index` is not assumed equal to env task_id, especially on
    `libero_90` where unique texts (73) do not match the 90-task benchmark.
    """

    if configured is not None:
        return configured
    require_libero_runtime()
    from libero.libero import benchmark

    from vla_fewshot.data.task_text import normalize_task_text

    benches = benchmark.get_benchmark_dict()
    if suite not in benches:
        raise RuntimeError(f"unknown LIBERO suite {suite!r}")
    wanted = normalize_task_text(task_text)
    matches = [
        index
        for index, task in enumerate(benches[suite]().tasks)
        if normalize_task_text(task.language) == wanted
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one LIBERO task matching {wanted!r} in {suite}, got {matches}"
        )
    return matches[0]


def run_libero_doctor_probe(
    *,
    suite: str = "libero_goal",
    task_id: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Reset, observe and step one hard-reset two-camera environment."""

    import gymnasium as gym
    import numpy as np
    import torch
    from lerobot.envs.libero import create_libero_envs
    from lerobot.processor.env_processor import LiberoProcessorStep

    envs = create_libero_envs(
        task=suite,
        n_envs=1,
        env_cls=gym.vector.SyncVectorEnv,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        control_mode="relative",
        gym_kwargs={
            "task_ids": [task_id],
            "obs_type": "pixels_agent_pos",
            "hard_reset": True,
            "observation_width": 256,
            "observation_height": 256,
        },
    )
    env = envs[suite][task_id]
    try:
        observation, _ = env.reset(seed=seed)
        pixels = observation["pixels"]
        required_cameras = {"image", "image2"}
        if set(pixels) != required_cameras:
            raise ValueError(
                f"expected raw cameras {sorted(required_cameras)}, got {sorted(pixels)}"
            )
        processor_input = {
            "observation.images.image": torch.as_tensor(pixels["image"]).permute(
                0, 3, 1, 2
            ),
            "observation.images.image2": torch.as_tensor(pixels["image2"]).permute(
                0, 3, 1, 2
            ),
            "observation.robot_state": _to_torch_tree(
                observation["robot_state"],
                torch,
            ),
        }
        processed = LiberoProcessorStep().observation(processor_input)
        canonical = apply_canonical_image_keys(processed)
        state = processed["observation.state"]
        if tuple(state.shape) != (1, 8):
            raise ValueError(f"expected state shape (1, 8), got {tuple(state.shape)}")
        action = np.zeros((1, 7), dtype=np.float32)
        action[:, -1] = -1.0
        next_observation, _, _, _, _ = env.step(action)
        return {
            "suite": suite,
            "task_id": task_id,
            "seed": seed,
            "raw_camera_keys": sorted(pixels),
            "canonical_keys": sorted(canonical),
            "main_shape": list(pixels["image"].shape),
            "wrist_shape": list(pixels["image2"].shape),
            "main_dtype": str(pixels["image"].dtype),
            "wrist_dtype": str(pixels["image2"].dtype),
            "processed_keys": sorted(processed),
            "state_shape": list(state.shape),
            "state_finite": bool(torch.isfinite(state).all().item()),
            "action_shape": list(action.shape),
            "step_camera_keys": sorted(next_observation["pixels"]),
            "control_mode": "relative",
            "hard_reset": True,
        }
    finally:
        env.close()


class LiberoRuntime:
    """Thin wrapper over pinned create_libero_envs for n_envs=1 replay."""

    def __init__(
        self,
        *,
        suite: str,
        task_id: int,
        seed: int = 0,
        control_mode: str = "relative",
        hard_reset: bool = True,
    ) -> None:
        require_libero_runtime()
        import gymnasium as gym
        from lerobot.envs.libero import create_libero_envs

        if control_mode != "relative":
            raise ValueError(f"unsupported control_mode {control_mode}")
        if not hard_reset:
            raise ValueError("hard_reset must stay true")
        self.suite = suite
        self.task_id = task_id
        self.seed = seed
        self._envs = create_libero_envs(
            task=suite,
            n_envs=1,
            env_cls=gym.vector.SyncVectorEnv,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            control_mode=control_mode,
            gym_kwargs={
                "task_ids": [task_id],
                "obs_type": "pixels_agent_pos",
                "hard_reset": True,
                "observation_width": 256,
                "observation_height": 256,
            },
        )
        self._env = self._envs[suite][task_id]
        self._closed = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self._env.reset(seed=self.seed if seed is None else seed)
        return self.policy_observation(observation), self._unbatch_info(info)

    def step(
        self, action: list[float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import numpy as np

        batched = np.asarray(action, dtype=np.float32).reshape(1, 7)
        observation, reward, terminated, truncated, info = self._env.step(batched)
        return (
            self.policy_observation(observation),
            float(np.asarray(reward).reshape(-1)[0]),
            bool(np.asarray(terminated).reshape(-1)[0]),
            bool(np.asarray(truncated).reshape(-1)[0]),
            self._unbatch_info(info),
        )

    def policy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        import torch
        from lerobot.processor.env_processor import LiberoProcessorStep

        pixels = observation["pixels"]
        processor_input = {
            "observation.images.image": torch.as_tensor(pixels["image"]).permute(
                0, 3, 1, 2
            ),
            "observation.images.image2": torch.as_tensor(pixels["image2"]).permute(
                0, 3, 1, 2
            ),
            "observation.robot_state": _to_torch_tree(observation["robot_state"], torch),
        }
        processed = LiberoProcessorStep().observation(processor_input)
        canonical = apply_canonical_image_keys(processed)
        state = processed["observation.state"]
        if hasattr(state, "detach"):
            canonical["observation.state"] = state.detach().cpu().reshape(-1).tolist()
        else:
            canonical["observation.state"] = flatten_libero_robot_state(
                observation["robot_state"]
            )
        return canonical

    def extract_main_hwc(self, observation: dict[str, Any]) -> Any:
        image = observation.get("observation.images.image")
        return _first_hwc(image)

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    def _unbatch_info(self, info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {"is_success": False}
        success = info.get("is_success", False)
        if isinstance(success, (list, tuple)):
            success = success[0] if success else False
        if hasattr(success, "reshape"):
            success = success.reshape(-1)[0]
        if hasattr(success, "item"):
            success = success.item()
        return {**info, "is_success": bool(success)}


def _first_hwc(image: Any) -> Any:
    if image is None:
        return None
    if hasattr(image, "permute") and getattr(image, "ndim", 0) == 4:
        tensor = image[0].permute(1, 2, 0)
        if tensor.dtype.is_floating_point:
            tensor = (tensor.clamp(0, 1) * 255).byte()
        return tensor.detach().cpu().numpy()
    if hasattr(image, "shape") and len(image.shape) == 4:
        return image[0]
    return image
