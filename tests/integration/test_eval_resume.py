import os
import subprocess
import sys
from pathlib import Path

from vla_fewshot.config import load_config
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.evaluation.runner import run_static_evaluation, static_smoke_config
from vla_fewshot.evaluation.store import RolloutStore


ROOT = Path(__file__).resolve().parents[2]


def test_interrupted_eval_resumes_without_duplicates(tmp_path: Path) -> None:
    config = static_smoke_config(load_config(ROOT / "configs" / "eval" / "final.yaml"))
    splits = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")
    output = tmp_path / "eval"
    first = run_static_evaluation(
        config=config,
        output_dir=output,
        checkpoint=tmp_path / "ckpt.txt",
        task_slug="bowl_stove",
        n_demos=5,
        train_seed=42,
        method="baseline",
        stage="target_eval",
        project_root=ROOT,
        splits=splits,
        max_new_rollouts=1,
    )
    assert first.written == 1
    assert first.complete is False
    second = run_static_evaluation(
        config=config,
        output_dir=output,
        checkpoint=tmp_path / "ckpt.txt",
        task_slug="bowl_stove",
        n_demos=5,
        train_seed=42,
        method="baseline",
        stage="target_eval",
        project_root=ROOT,
        splits=splits,
    )
    assert second.skipped == 1
    assert second.written == 2
    assert second.complete is True
    store = RolloutStore(output / "rollouts.jsonl")
    assert len(store) == 3
    keys = store.completed_keys()
    assert len(keys) == 3
    traces = list((output / "traces").glob("*.jsonl"))
    assert len(traces) == 3
    videos = list((output / "videos").glob("*"))
    assert videos
    failures = [
        record for record in store.records() if int(record["success"]) == 0
    ]
    for record in failures:
        assert record["video_uri"]
    successes = [record for record in store.records() if int(record["success"]) == 1]
    if successes:
        with_video = [record for record in successes if record.get("video_uri")]
        assert len(with_video) == 1


def test_language_control_pairs_share_fingerprint(tmp_path: Path) -> None:
    config = static_smoke_config(
        load_config(ROOT / "configs" / "eval" / "language_control.yaml")
    )
    result = run_static_evaluation(
        config=config,
        output_dir=tmp_path / "lang",
        checkpoint=tmp_path / "ckpt.txt",
        task_slug="bowl_stove",
        n_demos=0,
        train_seed=None,
        method="seen",
        stage="language_control",
        project_root=ROOT,
        language_control=True,
    )
    assert result.complete
    pairs = (tmp_path / "lang" / "language_pairs.json").read_text(encoding="utf-8")
    assert "action_l2_divergence" in pairs
    store = RolloutStore(tmp_path / "lang" / "rollouts.jsonl")
    assert len(store) == 6
    by_seed: dict[int, set[str]] = {}
    fingerprints: dict[int, set[str]] = {}
    for record in store.records():
        seed = int(record["eval_seed"])
        by_seed.setdefault(seed, set()).add(record["instruction_condition"])
        fingerprints.setdefault(seed, set()).add(record["initial_state_fingerprint"])
    assert all(value == {"correct", "wrong"} for value in by_seed.values())
    assert all(len(value) == 1 for value in fingerprints.values())


def test_eval_target_full_profile_fails_before_compute() -> None:
    env = os.environ.copy()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_target.py"),
            "--config",
            str(ROOT / "configs" / "eval" / "final.yaml"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
        env=env,
    )
    assert completed.returncode == 1
    assert "no GPU evaluation was started" in completed.stdout + completed.stderr
