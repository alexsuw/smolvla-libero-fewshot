from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest

from vla_fewshot.predictions import require_frozen_predictions
from vla_fewshot.storage.checksums import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def test_committed_predictions_match_frozen_lock() -> None:
    digest = require_frozen_predictions(root=ROOT)
    assert digest == sha256_file(ROOT / "predictions.md")
    assert len(digest) == 64


def test_edited_predictions_fail_closed_before_gpu(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "predictions.lock.yaml", tmp_path / "configs")
    shutil.copy(ROOT / "predictions.md", tmp_path / "predictions.md")
    require_frozen_predictions(root=tmp_path)
    (tmp_path / "predictions.md").write_text(
        (tmp_path / "predictions.md").read_text(encoding="utf-8") + "\n# edited\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no GPU training was started"):
        require_frozen_predictions(root=tmp_path)


def test_missing_predictions_lock_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "predictions.md").write_text("unlocked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no GPU training was started"):
        require_frozen_predictions(root=tmp_path)


def test_train_target_print_grid_does_not_require_lock() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_target.py"), "--print-grid"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert len([line for line in completed.stdout.splitlines() if line.strip()]) == 18


def test_train_target_refuses_before_gpu_when_lock_matches(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_target.py"),
            "--task",
            "drawer_middle",
            "--n-demos",
            "5",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path / "probe"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
        env=env,
    )
    assert completed.returncode == 1
    combined = completed.stdout + completed.stderr
    assert "no GPU training was started" in combined
    assert "predictions.md hash" not in combined
