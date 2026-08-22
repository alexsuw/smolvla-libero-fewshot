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

    Dataset `task_index` is not the env task_id. On `libero_goal` the NVIDIA
    parquet indices are 9/7/4 while the benchmark ids are 0/1/2. On
    `libero_90` 73 unique texts cover 90 BDDL tasks, so some languages match
    more than one env id; `configured` is only a disambiguator among those
    matches. When LIBERO cannot be imported, `configured` is returned as a
    CPU-test fallback.
    """

    try:
        require_libero_runtime()
    except RuntimeError:
        if configured is None:
            raise
        return configured

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
    if len(matches) == 1:
        return matches[0]
    if configured is not None and configured in matches:
        return configured
    raise RuntimeError(
        f"expected one LIBERO task matching {wanted!r} in {suite}, got {matches}"
    )
