import os
import subprocess
import sys
from pathlib import Path

from vla_fewshot.config import load_config
from vla_fewshot.training.compare import run_resume_compare_protocol


ROOT = Path(__file__).resolve().parents[2]


def test_fresh_process_resume_matches_continuous_run(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "train" / "smoke.yaml")
    report = run_resume_compare_protocol(
        config=config,
        output_dir=tmp_path / "compare",
        command=["python", "scripts/train_seen.py"],
        config_path=ROOT / "configs" / "train" / "smoke.yaml",
        project_root=ROOT,
        train_script=ROOT / "scripts" / "train_seen.py",
        log_freq=1,
        backup_dir=tmp_path / "backup",
    )
    assert report["passed"], report["checks"]
    assert report["fresh_process_returncode"] == 0
    assert (tmp_path / "backup" / "resume_compare.json").exists()
    assert (tmp_path / "compare" / "run_a" / "metrics.csv").exists()
    assert (tmp_path / "compare" / "run_a" / "events.jsonl").exists()
    assert (tmp_path / "compare" / "run_a" / "tensorboard" / "tags.jsonl").exists()
    assert (tmp_path / "compare" / "run_a" / "manifest.json").exists()


def test_train_seen_full_profile_fails_before_compute() -> None:
    env = os.environ.copy()
    env.pop("VLA_DATASETS_DIR", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_seen.py")],
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
