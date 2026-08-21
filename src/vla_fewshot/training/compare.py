"""Exact 0→200 vs 0→100→200 checkpoint comparison."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from vla_fewshot.config import TrainConfig
from vla_fewshot.logging.csv_logger import read_metric_column
from vla_fewshot.reproducibility import atomic_write_json, atomic_write_text
from vla_fewshot.storage.layout import (
    CHECKPOINT_OPTIMIZER_NAME,
    CHECKPOINT_TRAIN_STATE_NAME,
    CHECKPOINT_WEIGHTS_NAME,
    METRICS_CSV_NAME,
    step_directory,
)
from vla_fewshot.storage.sync import execute_local_mirror
from vla_fewshot.training.checkpoint import load_json, verify_checkpoint_dir
from vla_fewshot.training.trainer import run_static_training


def compare_checkpoints(
    path_a: Path,
    path_b: Path,
    *,
    run_a: Path | None = None,
    run_b: Path | None = None,
) -> dict[str, Any]:
    report_a = verify_checkpoint_dir(path_a)
    report_b = verify_checkpoint_dir(path_b)
    weights_a = load_json(path_a / CHECKPOINT_WEIGHTS_NAME)
    weights_b = load_json(path_b / CHECKPOINT_WEIGHTS_NAME)
    optim_a = load_json(path_a / CHECKPOINT_OPTIMIZER_NAME)
    optim_b = load_json(path_b / CHECKPOINT_OPTIMIZER_NAME)
    state_a = load_json(path_a / CHECKPOINT_TRAIN_STATE_NAME)
    state_b = load_json(path_b / CHECKPOINT_TRAIN_STATE_NAME)
    checks = {
        "global_step": state_a["global_step"] == state_b["global_step"],
        "sample_order": state_a["sample_order"] == state_b["sample_order"],
        "weights": weights_a == weights_b,
        "optimizer": optim_a == optim_b,
        "fresh_load_a": bool(report_a["fresh_load_verified"]),
        "fresh_load_b": bool(report_b["fresh_load_verified"]),
    }
    if run_a is not None and run_b is not None:
        losses_a = read_metric_column(run_a / METRICS_CSV_NAME, "loss")
        losses_b = read_metric_column(run_b / METRICS_CSV_NAME, "loss")
        checks["loss_curve"] = losses_a == losses_b
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "step_a": state_a["global_step"],
        "step_b": state_b["global_step"],
        "checkpoint_a": str(path_a),
        "checkpoint_b": str(path_b),
        "tolerance": "exact",
        "acceptance_complete": passed,
        "notes": (
            "Static CPU toy trainer proves the checkpoint/resume protocol. "
            "Full SmolVLA 200-step smoke still requires Linux CUDA."
        ),
    }


def run_resume_compare_protocol(
    *,
    config: TrainConfig,
    output_dir: Path,
    command: list[str],
    config_path: Path,
    project_root: Path,
    train_script: Path,
    log_freq: int = 1,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    """Run A: 0→200. Run B: 0→100 then a fresh process 100→200."""

    output_dir.mkdir(parents=True, exist_ok=True)
    run_a = output_dir / "run_a"
    run_b = output_dir / "run_b"
    existing = output_dir / "resume_compare.json"
    if run_a.exists() or run_b.exists() or existing.exists():
        if existing.is_file() and run_a.is_dir() and run_b.is_dir():
            report = load_json(existing)
            if report.get("passed") is True:
                return report
        raise FileExistsError(
            f"refusing to overwrite resume-compare runs under {output_dir}"
        )
    run_static_training(
        config=config,
        run_dir=run_a,
        command=command,
        config_path=config_path,
        project_root=project_root,
        profile="static",
        log_freq=log_freq,
        run_id="run_a",
    )
    run_static_training(
        config=config,
        run_dir=run_b,
        command=command,
        config_path=config_path,
        project_root=project_root,
        profile="static",
        stop_after=100,
        log_freq=log_freq,
        run_id="run_b",
    )
    resume_from = step_directory(run_b, 100)
    completed = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--config",
            str(config_path),
            "--profile",
            "static",
            "--protocol",
            "train",
            "--resume-from",
            str(resume_from),
            "--output-dir",
            str(run_b),
            "--log-freq",
            str(log_freq),
        ],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh-process resume failed: "
            f"exit={completed.returncode} stderr={completed.stderr!r} "
            f"stdout={completed.stdout!r}"
        )
    report = compare_checkpoints(
        step_directory(run_a, config.training.max_steps),
        step_directory(run_b, config.training.max_steps),
        run_a=run_a,
        run_b=run_b,
    )
    report["fresh_process_returncode"] = completed.returncode
    report["run_a"] = str(run_a)
    report["run_b"] = str(run_b)
    atomic_write_json(output_dir / "resume_compare.json", report, overwrite=True)
    lines = [
        "# M5 resume comparison (static)",
        "",
        f"- passed: `{report['passed']}`",
        f"- tolerance: `{report['tolerance']}`",
        f"- checks: `{report['checks']}`",
        "",
        report["notes"],
        "",
    ]
    atomic_write_text(
        output_dir / "resume_compare.md",
        "\n".join(lines),
        overwrite=True,
    )
    if backup_dir is not None:
        execute_local_mirror(output_dir, backup_dir, execute=True)
        report["backup_dir"] = str(backup_dir)
    return report
