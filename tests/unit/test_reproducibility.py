from pathlib import Path

import pytest

from vla_fewshot.config import PlatformConfig, RevisionsConfig, load_config
from vla_fewshot.paths import ProjectPaths
from vla_fewshot.reproducibility import (
    atomic_write_text,
    redact_text,
    write_environment_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def test_atomic_write_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_text(path, "first\n")
    with pytest.raises(FileExistsError):
        atomic_write_text(path, "second\n")
    assert path.read_text(encoding="utf-8") == "first\n"


def test_known_token_shapes_are_redacted() -> None:
    token = "hf_" + "A" * 32
    assert token not in redact_text(f"token={token}")


def test_tokenizers_parallelism_false_does_not_corrupt_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    payload = '{"terminated": false, "success": true}'
    assert redact_text(payload) == payload


def test_environment_bundle_contains_all_required_outputs(tmp_path: Path) -> None:
    platform = load_config(ROOT / "configs" / "platform" / "gpu_vm.yaml")
    revisions = load_config(ROOT / "configs" / "revisions.lock.yaml")
    assert isinstance(platform, PlatformConfig)
    assert isinstance(revisions, RevisionsConfig)
    paths = ProjectPaths(
        project_root=ROOT,
        data_root=tmp_path / "data",
        datasets_dir=tmp_path / "data" / "datasets",
        runs_dir=tmp_path / "data" / "runs",
        checkpoints_dir=tmp_path / "data" / "checkpoints",
        cache_dir=tmp_path / "data" / "cache",
        scratch_dir=tmp_path / "scratch",
        object_uri=None,
        durability_backend="persistent_disk",
        ephemeral=False,
    )

    files = write_environment_bundle(
        output_dir=tmp_path / "bundle",
        paths=paths,
        platform_config=platform,
        revisions=revisions,
        upstream_revisions={"acceptance_complete": True},
        command=["doctor", "--profile", "static"],
    )

    assert set(files) == {
        "environment_manifest",
        "pip_freeze",
        "system_info",
        "nvidia_smi",
        "ffmpeg_version",
        "upstream_revisions",
    }
    assert all(path.exists() for path in files.values())
