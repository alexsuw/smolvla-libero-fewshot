from pathlib import Path

import pytest

from vla_fewshot.data.layout import dataset_revision_root, resolve_datasets_dir


def test_revision_is_encoded_in_dataset_path(tmp_path: Path) -> None:
    revision = "e5907374380b8f96511957e6ba5582be52a1e179"
    root = dataset_revision_root(tmp_path, "nvidia/LIBERO_LeRobot_v3", revision)
    assert root.name == revision
    assert "nvidia_LIBERO_LeRobot_v3" in root.parts


def test_short_revision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="40-character SHA"):
        dataset_revision_root(tmp_path, "nvidia/LIBERO_LeRobot_v3", "abc")


def test_datasets_dir_requires_override_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLA_DATASETS_DIR", raising=False)
    with pytest.raises(RuntimeError, match="VLA_DATASETS_DIR"):
        resolve_datasets_dir()
