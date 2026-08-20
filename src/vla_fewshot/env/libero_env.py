"""Minimal pinned-LeRobot LIBERO probe used only by the M1 doctor."""

from __future__ import annotations

from typing import Any


def _to_torch_tree(value: Any, torch: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_torch_tree(child, torch) for key, child in value.items()}
    return torch.as_tensor(value)


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
            "main_shape": list(pixels["image"].shape),
            "wrist_shape": list(pixels["image2"].shape),
            "main_dtype": str(pixels["image"].dtype),
            "wrist_dtype": str(pixels["image2"].dtype),
            "processed_keys": sorted(processed),
            "state_shape": list(state.shape),
            "state_finite": bool(torch.isfinite(state).all().item()),
            "action_shape": list(action.shape),
            "step_camera_keys": sorted(next_observation["pixels"]),
        }
    finally:
        env.close()
