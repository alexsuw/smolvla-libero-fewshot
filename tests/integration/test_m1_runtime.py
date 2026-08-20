import os
from pathlib import Path

import pytest

from vla_fewshot.config import PlatformConfig, RevisionsConfig, load_config
from vla_fewshot.doctor import _check_ffmpeg, _check_mujoco_egl
from vla_fewshot.env.libero_env import run_libero_doctor_probe
from vla_fewshot.paths import resolve_paths
from vla_fewshot.storage.roundtrip import filesystem_roundtrip


ROOT = Path(__file__).resolve().parents[2]
RUN_GPU = os.environ.get("VLA_RUN_GPU_TESTS") == "1"
RUN_STORAGE = os.environ.get("VLA_RUN_STORAGE_TESTS") == "1"


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(not RUN_GPU, reason="set VLA_RUN_GPU_TESTS=1 on the GPU host")
def test_mujoco_egl_two_camera_and_av1() -> None:
    revisions = load_config(ROOT / "configs" / "revisions.lock.yaml")
    platform = load_config(ROOT / "configs" / "platform" / "gpu_vm.yaml")
    assert isinstance(revisions, RevisionsConfig)
    assert isinstance(platform, PlatformConfig)
    paths = resolve_paths(platform)

    assert _check_mujoco_egl()["render_shape"] == [32, 32, 3]
    assert _check_ffmpeg(revisions, paths.scratch_dir)["av1_roundtrip"]
    probe = run_libero_doctor_probe()
    assert probe["raw_camera_keys"] == ["image", "image2"]
    assert probe["state_shape"] == [1, 8]


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_STORAGE,
    reason="set VLA_RUN_STORAGE_TESTS=1 after mounting durable storage",
)
def test_configured_durable_storage_roundtrip() -> None:
    config_path = Path(
        os.environ.get(
            "VLA_PLATFORM_CONFIG",
            ROOT / "configs" / "platform" / "gpu_vm.yaml",
        )
    )
    platform = load_config(config_path)
    assert isinstance(platform, PlatformConfig)
    paths = resolve_paths(platform)
    assert not paths.ephemeral
    assert filesystem_roundtrip(paths.data_root).verified
