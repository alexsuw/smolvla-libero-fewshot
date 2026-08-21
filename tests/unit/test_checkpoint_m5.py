from pathlib import Path

import pytest

from vla_fewshot.config import load_config
from vla_fewshot.model.freezing import AllowlistError
from vla_fewshot.storage.layout import CHECKPOINT_COMPLETED_NAME, step_directory
from vla_fewshot.storage.retention import inventory_checkpoints
from vla_fewshot.storage.sync import execute_local_mirror
from vla_fewshot.training.checkpoint import CheckpointError, verify_checkpoint_dir
from vla_fewshot.training.resume import ResumeError, assert_resume_compatible
from vla_fewshot.training.trainer import run_static_training
from vla_fewshot.training.toy import ToyPolicy


ROOT = Path(__file__).resolve().parents[2]


def _smoke():
    return load_config(ROOT / "configs" / "train" / "smoke.yaml")


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "step_000050"
    directory.mkdir()
    (directory / "weights.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="COMPLETED.json"):
        verify_checkpoint_dir(directory)


def test_existing_run_directory_is_not_overwritten(tmp_path: Path) -> None:
    config = _smoke()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_static_training(
            config=config,
            run_dir=run_dir,
            command=["python", "scripts/train_seen.py"],
            config_path=ROOT / "configs" / "train" / "smoke.yaml",
            project_root=ROOT,
            stop_after=1,
        )


def test_resume_rejects_seed_change(tmp_path: Path) -> None:
    config = _smoke()
    run_dir = tmp_path / "run"
    run_static_training(
        config=config,
        run_dir=run_dir,
        command=["python", "scripts/train_seen.py"],
        config_path=ROOT / "configs" / "train" / "smoke.yaml",
        project_root=ROOT,
        stop_after=100,
    )
    other = config.model_copy(update={"training": config.training.model_copy(update={"seed": 123})})
    with pytest.raises(ResumeError, match="seed"):
        assert_resume_compatible(step_directory(run_dir, 100), other)


def test_sync_refuses_conflicting_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    source.mkdir()
    dest.mkdir()
    (source / "manifest.json").write_text('{"a": 1}\n', encoding="utf-8")
    (dest / "manifest.json").write_text('{"a": 2}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        execute_local_mirror(source, dest, execute=True)


def test_prune_never_deletes(tmp_path: Path) -> None:
    run = tmp_path / "run" / "checkpoints" / "step_000100"
    run.mkdir(parents=True)
    (run / CHECKPOINT_COMPLETED_NAME).write_text("{}\n", encoding="utf-8")
    report = inventory_checkpoints(tmp_path)
    assert report["delete_enabled"] is False
    assert (run / CHECKPOINT_COMPLETED_NAME).exists()
    assert all(item["action"] == "keep" for item in report["candidates"])


def test_missing_expert_fails_closed_before_named_optimizer() -> None:
    config = _smoke()
    policy = ToyPolicy(seed=42)
    del policy._params[
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"
    ]
    from vla_fewshot.model.freezing import assert_module_trainable_scope

    with pytest.raises(AllowlistError, match="allowlist"):
        assert_module_trainable_scope(policy, config.trainable_scope)
