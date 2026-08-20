from pathlib import Path

import pytest

from vla_fewshot.paths import resolve_paths


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
