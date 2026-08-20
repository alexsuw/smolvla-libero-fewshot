import os
from pathlib import Path

import pytest

from vla_fewshot.config import PlatformConfig, load_config
from vla_fewshot.paths import resolve_paths


ROOT = Path(__file__).resolve().parents[2]


PATH_VARIABLES = (
    "VLA_DATA_ROOT",
    "VLA_DATASETS_DIR",
    "VLA_RUNS_DIR",
    "VLA_CHECKPOINTS_DIR",
    "VLA_CACHE_DIR",
    "VLA_SCRATCH_DIR",
)


def test_paths_are_resolved_only_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLA_PROJECT_ROOT", str(tmp_path / "project"))
    for name in PATH_VARIABLES:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("VLA_OBJECT_URI", "s3://example/project")

    paths = resolve_paths()

    assert paths.runs_dir == (tmp_path / "vla_runs_dir").resolve()
    assert paths.object_uri == "s3://example/project"


def test_missing_runtime_root_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PATH_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="VLA_DATA_ROOT"):
        resolve_paths()


def test_gpu_vm_derives_subdirectories_from_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs" / "platform" / "gpu_vm.yaml")
    assert isinstance(config, PlatformConfig)
    monkeypatch.setenv("VLA_DATA_ROOT", str(tmp_path / "persistent"))
    for name in PATH_VARIABLES[1:]:
        monkeypatch.delenv(name, raising=False)

    paths = resolve_paths(config)

    assert paths.datasets_dir == (tmp_path / "persistent" / "datasets").resolve()
    assert paths.durability_backend == "persistent_disk"
    assert not paths.ephemeral


def test_colab_requires_actual_drive_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs" / "platform" / "colab.yaml")
    assert isinstance(config, PlatformConfig)
    mount = tmp_path / "drive"
    mount.mkdir()
    storage = config.storage.model_copy(
        update={
            "data_root_default": str(mount / "MyDrive" / "vla-fewshot"),
            "drive_mount_root": str(mount),
        }
    )
    config = config.model_copy(update={"storage": storage})
    monkeypatch.delenv("VLA_DATA_ROOT", raising=False)
    monkeypatch.setattr(os.path, "ismount", lambda path: Path(path) == mount)

    paths = resolve_paths(config)

    assert paths.durability_backend == "google_drive"
    assert not paths.ephemeral
