from pathlib import Path
import json
import subprocess
import sys

import pytest

from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.evaluation.baseline_eval import (
    BaselineEvalError,
    baseline_eval_commands,
    check_baseline_eval_records,
    verify_baseline_run_eval,
)
from vla_fewshot.storage.layout import (
    CHECKPOINT_CHECKSUMS_NAME,
    CHECKPOINT_COMPLETED_NAME,
    CHECKPOINT_OPTIMIZER_NAME,
    CHECKPOINT_RNG_NAME,
    CHECKPOINT_TRAIN_STATE_NAME,
    CHECKPOINT_WEIGHTS_NAME,
    MANIFEST_NAME,
    step_directory,
)


ROOT = Path(__file__).resolve().parents[2]
SPLITS = load_target_splits(ROOT / "configs" / "splits" / "target_splits.json")


def _record(*, seed: int, success: int, video: str | None, n_demos: int = 5) -> dict:
    return {
        "protocol_id": "final_v1",
        "method": "baseline",
        "stage": "target_eval",
        "n_demos": n_demos,
        "train_seed": 42,
        "task_slug": "bowl_stove",
        "training_episode_ids": [13, 15, 16, 22, 36],
        "eval_seed": seed,
        "success": success,
        "trace_uri": f"traces/{seed}.jsonl",
        "video_uri": video,
        "checkpoint_sha256": "abc",
        "instruction_condition": "correct",
    }


def _write_complete_toy_ckpt(run_dir: Path, step: int) -> None:
    directory = step_directory(run_dir, step)
    directory.mkdir(parents=True)
    for name in (
        CHECKPOINT_COMPLETED_NAME,
        CHECKPOINT_CHECKSUMS_NAME,
        CHECKPOINT_RNG_NAME,
        CHECKPOINT_TRAIN_STATE_NAME,
        CHECKPOINT_WEIGHTS_NAME,
        CHECKPOINT_OPTIMIZER_NAME,
    ):
        (directory / name).write_text("{}\n", encoding="utf-8")


def test_print_grid_lists_eighteen_eval_commands() -> None:
    commands = baseline_eval_commands()
    assert len(commands) == 18
    assert "--run-dir" in commands[0]
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_target.py"), "--print-grid"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 18
    assert "--task drawer_middle --n-demos 5 --seed 42" in lines[0]


def test_verify_script_print_grid() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_baseline_eval.py"), "--print-grid"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert len([line for line in completed.stdout.splitlines() if line.strip()]) == 18


def test_failure_without_video_is_rejected() -> None:
    records = [
        _record(seed=1000, success=0, video=None),
        _record(seed=1001, success=1, video="ok.ppm"),
    ]
    with pytest.raises(BaselineEvalError, match="failures missing video"):
        check_baseline_eval_records(
            records,
            task_slug="bowl_stove",
            n_demos=5,
            train_seed=42,
            episode_ids=[13, 15, 16, 22, 36],
            min_rollouts=2,
        )


def test_complete_checkpoint_eval_is_accepted(tmp_path: Path) -> None:
    train = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    _write_complete_toy_ckpt(train, 3000)
    (train / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "stage": "target",
                "method": "baseline",
                "task_slug": "bowl_stove",
                "n_demos": 5,
                "train_seed": 42,
            }
        ),
        encoding="utf-8",
    )
    jsonl = eval_dir / "step_003000" / "bowl_stove" / "rollouts.jsonl"
    jsonl.parent.mkdir(parents=True)
    records = [
        _record(seed=1000, success=0, video="fail.ppm"),
        _record(seed=1001, success=1, video="ok.ppm"),
    ]
    jsonl.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    report = verify_baseline_run_eval(
        train,
        eval_dir,
        task_slug="bowl_stove",
        n_demos=5,
        train_seed=42,
        splits=SPLITS,
        min_rollouts=2,
    )
    assert report["complete"]
    assert report["n_checkpoints"] == 1


def test_missing_eval_jsonl_fails(tmp_path: Path) -> None:
    train = tmp_path / "train"
    _write_complete_toy_ckpt(train, 3000)
    with pytest.raises(BaselineEvalError, match="missing eval rollouts"):
        verify_baseline_run_eval(
            train,
            tmp_path / "eval",
            task_slug="bowl_stove",
            n_demos=5,
            train_seed=42,
            splits=SPLITS,
            min_rollouts=2,
        )
