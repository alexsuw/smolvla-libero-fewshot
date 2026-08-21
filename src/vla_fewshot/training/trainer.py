"""CPU smoke trainer with allowlist-before-optimizer and exact resume."""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.config import TrainConfig
from vla_fewshot.logging.csv_logger import CsvMetricsLogger, metrics_row
from vla_fewshot.logging.events import JsonlEventLogger
from vla_fewshot.logging.manifest import (
    mark_completed,
    mark_failed,
    mark_interrupted,
    new_training_manifest,
    update_manifest,
    write_manifest,
    write_resolved_config,
)
from vla_fewshot.logging.tensorboard import TensorBoardLogger
from vla_fewshot.model.freezing import assert_module_trainable_scope
from vla_fewshot.reproducibility import atomic_write_json
from vla_fewshot.storage.layout import (
    ENVIRONMENT_MANIFEST_NAME,
    EVENTS_JSONL_NAME,
    METRICS_CSV_NAME,
    RESOLVED_CONFIG_NAME,
    TENSORBOARD_DIRNAME,
    TRAIN_LOG_NAME,
    TRAINABLE_PARAMETERS_NAME,
    run_lock_path,
)
from vla_fewshot.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    train_state_payload,
)
from vla_fewshot.training.optim import (
    ToyAdamW,
    current_lr,
    resolve_gradient_accumulation,
)
from vla_fewshot.training.replay_mixer import assert_replay_disabled
from vla_fewshot.training.resume import assert_resume_compatible
from vla_fewshot.training.sampler import DeterministicSampler
from vla_fewshot.training.toy import ToyPolicy


class TrainError(RuntimeError):
    """Raised for fail-closed training contract violations."""


@dataclass
class TrainResult:
    run_dir: Path
    global_step: int
    status: str
    final_checkpoint: Path | None
    last_loss: float | None


def _disable_wandb() -> None:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    if "wandb" in __import__("sys").modules:
        raise TrainError("wandb must not be imported by the training stack")


def _should_save(step: int, config: TrainConfig) -> bool:
    if step in config.checkpoint.save_steps:
        return True
    if config.checkpoint.every_steps and step % config.checkpoint.every_steps == 0:
        return True
    if step in config.checkpoint.milestones:
        return True
    return False


def _append_log(run_dir: Path, message: str) -> None:
    path = run_dir / TRAIN_LOG_NAME
    line = f"{datetime.now(UTC).isoformat()} {message}\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_lock(run_dir: Path) -> Path:
    lock = run_lock_path(run_dir)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise TrainError(f"run lock exists: {lock}") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def prepare_static_modules(
    config: TrainConfig,
    *,
    output_dir: Path,
) -> tuple[ToyPolicy, ToyAdamW, DeterministicSampler, dict[str, Any]]:
    """Fail-closed allowlist, then optimizer. Never reverse that order."""

    assert_replay_disabled(config)
    if config.tracking.wandb_enabled:
        raise TrainError("tracking.wandb_enabled must stay false")
    policy = ToyPolicy(seed=config.training.seed)
    report = assert_module_trainable_scope(policy, config.trainable_scope, output_dir=output_dir)
    optimizer = ToyAdamW(policy, config.optimizer)
    sampler = DeterministicSampler(seed=config.training.seed)
    return policy, optimizer, sampler, report


def run_static_training(
    *,
    config: TrainConfig,
    run_dir: Path,
    command: list[str],
    config_path: Path,
    project_root: Path,
    profile: str = "static",
    resume_from: Path | None = None,
    stop_after: int | None = None,
    log_freq: int = 1,
    install_signal_handlers: bool = False,
    run_id: str | None = None,
) -> TrainResult:
    _disable_wandb()
    accumulation = resolve_gradient_accumulation(config.training)
    stop_at = stop_after or config.training.max_steps
    if stop_at < 1:
        raise TrainError("stop_after must be positive")

    resume = resume_from is not None
    if resume:
        if resume_from is None or not resume_from.exists():
            raise TrainError("resume-from path does not exist")
        assert_resume_compatible(resume_from, config)
        if not run_dir.exists():
            raise TrainError("resume requires the existing run directory")
    else:
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run directory {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        write_resolved_config(run_dir / RESOLVED_CONFIG_NAME, config)
        manifest = new_training_manifest(
            run_id=run_id or run_dir.name,
            config=config,
            command=command,
            project_root=project_root,
            config_path=config_path,
            profile=profile,
        )
        write_manifest(run_dir, manifest)
        atomic_write_json(
            run_dir / ENVIRONMENT_MANIFEST_NAME,
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "profile": profile,
                "wandb_mode": os.environ.get("WANDB_MODE", "disabled"),
                "command": command,
            },
        )

    lock = _acquire_lock(run_dir)
    csv_logger = CsvMetricsLogger(run_dir / METRICS_CSV_NAME)
    events = JsonlEventLogger(run_dir / EVENTS_JSONL_NAME)
    tb = TensorBoardLogger(run_dir / TENSORBOARD_DIRNAME)
    stop_requested = {"value": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        stop_requested["value"] = True
        _append_log(run_dir, f"signal {signum} requested checkpoint and stop")

    previous_handlers: dict[int, Any] = {}
    if install_signal_handlers:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, _on_signal)

    started = time.perf_counter()
    last_loss: float | None = None
    final_checkpoint: Path | None = None
    global_step = 0
    samples_seen = 0
    accum_position = 0
    try:
        policy, optimizer, sampler, scope_report = prepare_static_modules(
            config, output_dir=run_dir
        )
        update_manifest(
            run_dir,
            trainable_parameter_count=scope_report["trainable_parameters"],
            total_parameter_count=scope_report["total_parameters"],
        )
        if resume:
            loaded = load_checkpoint(
                resume_from,  # type: ignore[arg-type]
                policy=policy,
                optimizer=optimizer,
                sampler=sampler,
            )
            train_state = loaded["train_state"]
            global_step = int(train_state["global_step"])
            samples_seen = int(train_state["samples_seen"])
            accum_position = int(train_state["accumulation_position"])
            events.emit("resume", {"global_step": global_step, "path": str(resume_from)})
            _append_log(run_dir, f"resumed from {resume_from} at step {global_step}")
        else:
            events.emit("run_start", {"run_id": run_dir.name, "profile": profile})
            _append_log(run_dir, "static smoke training start")

        if not (run_dir / TRAINABLE_PARAMETERS_NAME).exists():
            raise TrainError("trainable_parameters.txt missing before optimizer steps")

        n_samples = sampler.n_samples
        while global_step < stop_at:
            if accum_position == 0:
                policy.zero_grad()
            data_t0 = time.perf_counter()
            sample = sampler.next_sample()
            data_time = time.perf_counter() - data_t0
            step_t0 = time.perf_counter()
            if accum_position == 0:
                window_loss = 0.0
            loss = policy.forward_loss(sample.x, sample.y)
            window_loss += loss
            accum_position += 1
            samples_seen += 1
            if accum_position < accumulation:
                continue
            mean_loss = window_loss / float(accumulation)
            last_loss = mean_loss
            policy.scale_grads(1.0 / float(accumulation))
            grad_norm = optimizer.clip_grad_norm(config.training.max_grad_norm)
            lr = current_lr(
                config.scheduler,
                config.optimizer,
                global_step,
                config.training.max_steps,
            )
            optimizer.step(lr)
            global_step += 1
            accum_position = 0
            step_time = time.perf_counter() - step_t0
            elapsed = time.perf_counter() - started
            epoch_fraction = samples_seen / float(n_samples)
            sps = (config.training.effective_batch_size / step_time) if step_time else 0.0
            if log_freq > 0 and global_step % log_freq == 0:
                row = metrics_row(
                    elapsed_seconds=elapsed,
                    global_step=global_step,
                    samples_seen=samples_seen,
                    epoch_fraction=epoch_fraction,
                    loss=mean_loss,
                    learning_rate=lr,
                    grad_norm=grad_norm,
                    samples_per_second=sps,
                    data_time_seconds=data_time,
                    step_time_seconds=step_time,
                )
                csv_logger.append(row)
                events.emit(
                    "step",
                    {"global_step": global_step, "loss": mean_loss, "lr": lr},
                )
                tb.log_train_step(
                    step=global_step,
                    loss=mean_loss,
                    learning_rate=lr,
                    grad_norm=grad_norm,
                    samples_per_second=sps,
                )
                _append_log(
                    run_dir, f"step={global_step} loss={mean_loss:.8f} lr={lr:.8g}"
                )
            if _should_save(global_step, config) or stop_requested["value"]:
                state = train_state_payload(
                    global_step=global_step,
                    samples_seen=samples_seen,
                    accumulation_position=accum_position,
                    epoch_fraction=epoch_fraction,
                    metrics_cursor=csv_logger.row_count(),
                    sampler=sampler,
                    sample_order=sampler.order_hash_payload(),
                )
                names = (run_dir / TRAINABLE_PARAMETERS_NAME).read_text(encoding="utf-8").splitlines()
                final_checkpoint = save_checkpoint(
                    run_dir,
                    step=global_step,
                    config=config,
                    policy=policy,
                    optimizer=optimizer,
                    sampler=sampler,
                    train_state=state,
                    trainable_names=names,
                )
                events.emit("checkpoint_saved", {"path": str(final_checkpoint), "step": global_step})
                _append_log(run_dir, f"saved {final_checkpoint}")
            if stop_requested["value"]:
                break

        tb.close()
        if stop_requested["value"]:
            mark_interrupted(run_dir)
            return TrainResult(run_dir, global_step, "interrupted", final_checkpoint, last_loss)
        if global_step >= config.training.max_steps and final_checkpoint is not None:
            mark_completed(run_dir, final_checkpoint_uri=str(final_checkpoint))
            events.emit("run_completed", {"global_step": global_step})
            status = "completed"
        else:
            update_manifest(run_dir, status="stopped")
            status = "stopped"
        return TrainResult(run_dir, global_step, status, final_checkpoint, last_loss)
    except Exception as error:
        tb.close()
        mark_failed(run_dir, error)
        events.emit("run_failed", {"type": type(error).__name__})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if lock.exists():
            lock.unlink()
