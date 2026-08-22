"""Project-owned SmolVLA training loop. Never calls lerobot-train."""

from __future__ import annotations

import os
import random
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla_fewshot.config import TrainConfig
from vla_fewshot.data.layout import dataset_revision_root, suite_root
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
from vla_fewshot.model.freezing import apply_trainable_scope, assert_module_trainable_scope
from vla_fewshot.model.peft import wrap_policy_lora
from vla_fewshot.model.smolvla import load_pinned_smolvla
from vla_fewshot.reproducibility import atomic_write_json
from vla_fewshot.storage.layout import (
    ENVIRONMENT_MANIFEST_NAME,
    EVENTS_JSONL_NAME,
    METRICS_CSV_NAME,
    RESOLVED_CONFIG_NAME,
    TENSORBOARD_DIRNAME,
    TRAINABLE_PARAMETERS_NAME,
)
from vla_fewshot.training.autofit import fit_physical_batch, try_smolvla_minibatch
from vla_fewshot.training.batching import is_cuda_oom, resolve_training_batch
from vla_fewshot.training.checkpoint import train_state_payload
from vla_fewshot.training.cursor import FrameCursor
from vla_fewshot.training.data import (
    action_delta_timestamps,
    assert_suite_videos,
    load_suite_for_train,
)
from vla_fewshot.training.full import refuse_lerobot_train_cli, require_full_training_runtime
from vla_fewshot.training.optim import current_lr, resolve_gradient_accumulation
from vla_fewshot.training.precision import autocast_cm, cuda_bf16_supported, resolve_precision
from vla_fewshot.training.replay_mixer import (
    ReplayMixer,
    assert_replay_disabled,
    assert_replay_pool,
    gather_mixed_samples,
)
from vla_fewshot.training.resume import assert_resume_compatible
from vla_fewshot.training.baseline import cap_optimizer_steps
from vla_fewshot.training.stats import (
    collect_state_action_rows,
    mean_std,
    overlay_state_action_stats,
)
from vla_fewshot.training.torch_checkpoint import (
    load_policy_weights,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from vla_fewshot.training.trainer import (
    TrainError,
    TrainResult,
    _acquire_lock,
    _append_log,
    _disable_wandb,
    _should_save,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        first = values[0]
        if torch.is_tensor(first):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch


def _fetch_samples(dataset: Any, indices: list[int]) -> list[dict[str, Any]]:
    """Decode one ordered batch; safe to run in a single prefetch thread."""

    return [dataset[index] for index in indices]


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if hasattr(value, "to") else value
    return moved


def _make_preprocessor(policy: Any, stats: dict[str, Any], device: str) -> Any:
    from lerobot.policies.factory import make_pre_post_processors

    try:
        preprocessor, _post = make_pre_post_processors(
            policy.config,
            pretrained_path=None,
            dataset_stats=stats,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    except TypeError:
        preprocessor, _post = make_pre_post_processors(
            policy.config,
            pretrained_path=None,
            dataset_stats=stats,
        )
    return preprocessor


def _load_lerobot_dataset(
    *,
    config: TrainConfig,
    revision_root: Path,
    episode_ids: list[int] | None,
    chunk_size: int,
    fps: float,
    suite: str | None = None,
) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    suite_dir = suite_root(revision_root, suite or config.dataset.suite)
    assert_suite_videos(suite_dir)
    dataset = LeRobotDataset(
        repo_id=config.dataset.repo_id,
        root=suite_dir,
        episodes=episode_ids,
        revision=config.dataset.revision,
        download_videos=False,
        delta_timestamps=action_delta_timestamps(fps=fps, chunk_size=chunk_size),
    )
    if len(dataset) < 1:
        raise TrainError("training dataset is empty")
    return dataset


def _adamw(policy: Any, config: TrainConfig) -> Any:
    import torch

    params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not params:
        raise TrainError("optimizer has no trainable parameters")
    return torch.optim.AdamW(
        params,
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


def prepare_full_training(
    config: TrainConfig,
    *,
    datasets_dir: Path,
    origin_checkpoint: Path | None = None,
    origin_sha256: str | None = None,
    episode_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Load policy/data and freeze physical batch. No run directory yet."""

    require_full_training_runtime()
    refuse_lerobot_train_cli()
    _disable_wandb()
    assert_replay_disabled(config)
    if config.peft is not None and (
        config.stage != "target" or config.method not in {"lora", "replay_lora"}
    ):
        raise TrainError("this training path does not wrap LoRA. no GPU training was started.")
    if config.tracking.wandb_enabled:
        raise TrainError("tracking.wandb_enabled must stay false")
    if config.stage == "target":
        if origin_checkpoint is None:
            raise TrainError(
                "target training requires the frozen seen checkpoint. "
                "no GPU training was started."
            )
        if not episode_ids:
            raise TrainError(
                "target training requires explicit selected episode IDs. "
                "no GPU training was started."
            )
    _seed_everything(config.training.seed)

    import torch

    revision_root = dataset_revision_root(
        datasets_dir, config.dataset.repo_id, config.dataset.revision
    )
    meta, discovered_ids = load_suite_for_train(revision_root, config)
    resolved_ids = discovered_ids if episode_ids is None else list(episode_ids)
    fps = float(meta.fps or 20)
    loaded = load_pinned_smolvla(
        repo_id=config.model.repo_id,
        revision=config.model.revision,
        scope=config.trainable_scope,
        device="cuda",
    )
    policy = loaded["policy"]
    if origin_checkpoint is not None:
        load_policy_weights(
            origin_checkpoint, policy=policy, expected_sha256=origin_sha256
        )
    if config.peft is not None:
        policy = wrap_policy_lora(policy, config)
        loaded["policy"] = policy
    apply_trainable_scope(
        policy,
        config.trainable_scope,
        peft_enabled=config.peft is not None,
    )
    chunk_size = int(getattr(policy.config, "chunk_size", 50))
    dataset = _load_lerobot_dataset(
        config=config,
        revision_root=revision_root,
        episode_ids=resolved_ids,
        chunk_size=chunk_size,
        fps=fps,
    )
    replay_dataset = None
    if config.method == "replay_lora":
        from vla_fewshot.data.expected import SEEN_SUITE
        from vla_fewshot.data.metadata import load_suite_metadata

        replay_meta = load_suite_metadata(revision_root, SEEN_SUITE)
        assert_replay_pool(suite=replay_meta.suite, task_texts=list(replay_meta.unique_task_texts))
        replay_dataset = _load_lerobot_dataset(
            config=config,
            revision_root=revision_root,
            episode_ids=None,
            chunk_size=chunk_size,
            fps=float(replay_meta.fps or fps),
            suite=SEEN_SUITE,
        )
    stats = getattr(getattr(dataset, "meta", None), "stats", None) or meta.stats
    if not stats:
        raise TrainError("dataset stats.json is required for MEAN_STD training; identity smoke stats are forbidden")
    if episode_ids is not None:
        states, actions = collect_state_action_rows(
            dataset[index] for index in range(len(dataset))
        )
        stats = overlay_state_action_stats(
            stats, state=mean_std(states), action=mean_std(actions)
        )
    device = loaded["device"]
    preprocessor = _make_preprocessor(policy, stats, device)
    precision = resolve_precision(
        config.training.mixed_precision, cuda_bf16=cuda_bf16_supported()
    )
    scratch_opt = _adamw(policy, config)

    def _try(physical: int) -> None:
        batch = _move_batch(
            preprocessor(_collate([dataset[index % len(dataset)] for index in range(physical)])),
            device,
        )
        try_smolvla_minibatch(
            policy=policy, optimizer=scratch_opt, batch=batch, precision=precision
        )

    if config.training.physical_batch_size == "auto_fit":
        resolved = fit_physical_batch(config, try_batch=_try)
    else:
        resolved = resolve_training_batch(config)
        _try(int(resolved.training.physical_batch_size))

    del scratch_opt
    torch.cuda.empty_cache()
    return {
        "config": resolved,
        "policy": policy,
        "dataset": dataset,
        "preprocessor": preprocessor,
        "device": device,
        "precision": precision,
        "episode_ids": resolved_ids,
        "chunk_size": chunk_size,
        "fps": fps,
        "loaded": loaded,
        "n_samples": len(dataset),
        "origin_checkpoint": origin_checkpoint,
        "origin_sha256": origin_sha256,
        "replay_dataset": replay_dataset,
    }


def run_full_training(
    *,
    config: TrainConfig,
    run_dir: Path,
    command: list[str],
    config_path: Path,
    project_root: Path,
    datasets_dir: Path,
    resume_from: Path | None = None,
    stop_after: int | None = None,
    log_freq: int = 1,
    install_signal_handlers: bool = True,
    run_id: str | None = None,
    prepared: dict[str, Any] | None = None,
    origin_checkpoint: Path | None = None,
    origin_sha256: str | None = None,
    episode_ids: list[int] | None = None,
) -> TrainResult:
    """Train SmolVLA with the project checkpoint/logger stack."""

    bundle = prepared or prepare_full_training(
        config,
        datasets_dir=datasets_dir,
        origin_checkpoint=origin_checkpoint,
        origin_sha256=origin_sha256,
        episode_ids=episode_ids,
    )
    config = bundle["config"]
    policy = bundle["policy"]
    dataset = bundle["dataset"]
    preprocessor = bundle["preprocessor"]
    device = bundle["device"]
    precision = bundle["precision"]
    n_samples = int(bundle["n_samples"])
    replay_dataset = bundle.get("replay_dataset")
    mixer: ReplayMixer | None = None
    if config.method == "replay_lora":
        if replay_dataset is None:
            raise TrainError("Replay-LoRA is missing the libero_90 pool")
        if config.replay is None:
            raise TrainError("Replay-LoRA is missing replay config")
        mixer = ReplayMixer(
            n_target=n_samples,
            n_replay=len(replay_dataset),
            target_fraction=config.replay.target_fraction,
            seen_fraction=config.replay.seen_fraction,
            seed=config.training.seed,
            with_replacement=config.training.sample_with_replacement,
        )
    accumulation = resolve_gradient_accumulation(config.training)
    physical = int(config.training.physical_batch_size)
    prefetch_enabled = (
        mixer is None
        and accumulation == 1
        and config.training.num_workers > 0
    )
    resolved_cap = cap_optimizer_steps(
        max_steps=config.training.max_steps,
        epochs=config.training.epochs,
        n_samples=n_samples,
        effective_batch_size=int(config.training.effective_batch_size),
    )
    stop_at = resolved_cap if stop_after is None else min(int(stop_after), resolved_cap)
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
            profile="full",
        )
        manifest["resolved_precision"] = precision
        manifest["episode_ids"] = bundle["episode_ids"] or []
        if bundle.get("origin_checkpoint") is not None:
            manifest["base_checkpoint_uri"] = str(bundle["origin_checkpoint"])
            manifest["base_checkpoint_sha256"] = bundle.get("origin_sha256")
            manifest["origin_checkpoint_uri"] = str(bundle["origin_checkpoint"])
        if mixer is not None:
            manifest["replay_suite"] = "libero_90"
            manifest["replay_target_fraction"] = config.replay.target_fraction if config.replay else 0.75
            manifest["replay_seen_fraction"] = config.replay.seen_fraction if config.replay else 0.25
        if bundle.get("task_slug"):
            manifest["task_slug"] = bundle["task_slug"]
            manifest["task_text"] = bundle.get("task_text")
            manifest["n_demos"] = bundle.get("n_demos")
        write_manifest(run_dir, manifest)
        atomic_write_json(
            run_dir / ENVIRONMENT_MANIFEST_NAME,
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "profile": "full",
                "precision": precision,
                "physical_batch_size": physical,
                "gradient_accumulation": accumulation,
                "chunk_size": bundle["chunk_size"],
                "num_workers_config": config.training.num_workers,
                "num_workers_used": 1 if prefetch_enabled else 0,
                "ordered_batch_prefetch": prefetch_enabled,
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

    import torch

    last_loss: float | None = None
    final_checkpoint: Path | None = None
    last_saved_step = 0
    global_step = 0
    samples_seen = 0
    accum_position = 0
    cursor = FrameCursor.create(
        n_samples, seed=config.training.seed, with_replacement=config.training.sample_with_replacement
    )
    sampler: FrameCursor | ReplayMixer = mixer if mixer is not None else cursor
    prefetch_executor: ThreadPoolExecutor | None = None
    pending_indices: list[int] | None = None
    pending_future: Future[list[dict[str, Any]]] | None = None
    try:
        scope_report = assert_module_trainable_scope(
            policy,
            config.trainable_scope,
            output_dir=run_dir,
            peft_enabled=config.peft is not None,
        )
        optimizer = _adamw(policy, config)
        scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None

        def _persist_checkpoint() -> Path:
            nonlocal final_checkpoint, last_saved_step
            epoch_fraction = samples_seen / float(n_samples)
            state = train_state_payload(
                global_step=global_step,
                samples_seen=samples_seen,
                accumulation_position=accum_position,
                epoch_fraction=epoch_fraction,
                metrics_cursor=csv_logger.row_count(),
                sampler=sampler,  # type: ignore[arg-type]
                sample_order=list(cursor.order) if mixer is None else [],
            )
            state["resolved_precision"] = precision
            state["prefetched_indices"] = list(pending_indices or [])
            names = (run_dir / TRAINABLE_PARAMETERS_NAME).read_text(encoding="utf-8").splitlines()
            final_checkpoint = save_torch_checkpoint(
                run_dir,
                step=global_step,
                config=config,
                policy=policy,
                optimizer=optimizer,
                train_state=state,
                trainable_names=names,
                scaler=scaler,
            )
            last_saved_step = global_step
            events.emit(
                "checkpoint_saved",
                {"path": str(final_checkpoint), "step": global_step},
            )
            _append_log(run_dir, f"saved {final_checkpoint}")
            return final_checkpoint

        update_manifest(
            run_dir,
            trainable_parameter_count=scope_report["trainable_parameters"],
            total_parameter_count=scope_report["total_parameters"],
            resolved_precision=precision,
        )
        if resume:
            loaded_ckpt = load_torch_checkpoint(
                resume_from,  # type: ignore[arg-type]
                policy=policy,
                optimizer=optimizer,
                scaler=scaler,
            )
            train_state = loaded_ckpt["train_state"]
            global_step = int(train_state["global_step"])
            samples_seen = int(train_state["samples_seen"])
            accum_position = int(train_state["accumulation_position"])
            sampler_state = train_state["sampler"]
            if mixer is not None:
                mixer.load_state_dict(sampler_state)
            else:
                cursor.load_state_dict(sampler_state)
                restored = train_state.get("prefetched_indices", [])
                if restored:
                    pending_indices = [int(index) for index in restored]
            events.emit("resume", {"global_step": global_step, "path": str(resume_from)})
            _append_log(run_dir, f"resumed from {resume_from} at step {global_step}")
        else:
            events.emit("run_start", {"run_id": run_dir.name, "profile": "full", "precision": precision})
            _append_log(
                run_dir,
                f"full SmolVLA training start precision={precision} "
                f"physical={physical} accum={accumulation}",
            )

        if prefetch_enabled:
            prefetch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="vla-ordered-prefetch",
            )
            if pending_indices is None:
                pending_indices = cursor.next_indices(physical)
            pending_future = prefetch_executor.submit(
                _fetch_samples,
                dataset,
                list(pending_indices),
            )

        if not (run_dir / TRAINABLE_PARAMETERS_NAME).exists():
            raise TrainError("trainable_parameters.txt missing before optimizer steps")

        policy.train()
        started = time.perf_counter()
        window_loss = 0.0
        window_n_target = 0
        window_n_replay = 0
        while global_step < stop_at:
            try:
                if accum_position == 0:
                    optimizer.zero_grad(set_to_none=True)
                    window_loss = 0.0
                    window_n_target = 0
                    window_n_replay = 0
                data_t0 = time.perf_counter()
                mix_stats = {"target_fraction": 1.0, "seen_fraction": 0.0, "n_target": physical, "n_replay": 0}
                if mixer is not None:
                    draw = mixer.next_draw(physical)
                    samples = gather_mixed_samples(
                        draw, target_dataset=dataset, replay_dataset=replay_dataset
                    )
                    mix_stats = {
                        "target_fraction": draw.target_fraction,
                        "seen_fraction": draw.seen_fraction,
                        "n_target": draw.n_target,
                        "n_replay": draw.n_replay,
                    }
                elif prefetch_enabled:
                    if pending_future is None or pending_indices is None:
                        raise TrainError("ordered prefetch lost its pending batch")
                    samples = pending_future.result()
                    pending_future = None
                    pending_indices = None
                    if global_step + 1 < stop_at and not stop_requested["value"]:
                        pending_indices = cursor.next_indices(physical)
                        assert prefetch_executor is not None
                        pending_future = prefetch_executor.submit(
                            _fetch_samples,
                            dataset,
                            list(pending_indices),
                        )
                else:
                    indices = cursor.next_indices(physical)
                    samples = _fetch_samples(dataset, indices)
                window_n_target += int(mix_stats["n_target"])
                window_n_replay += int(mix_stats["n_replay"])
                batch = _move_batch(preprocessor(_collate(samples)), device)
                data_time = time.perf_counter() - data_t0
                step_t0 = time.perf_counter()
                with autocast_cm(precision):
                    loss, _details = policy.forward(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {global_step}")
                scaled = loss / float(accumulation)
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                window_loss += float(loss.detach().item())
                accum_position += 1
                samples_seen += physical
                if accum_position < accumulation:
                    continue
                mean_loss = window_loss / float(accumulation)
                last_loss = mean_loss
                lr = current_lr(
                    config.scheduler, config.optimizer, global_step, config.training.max_steps
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in policy.parameters() if parameter.requires_grad],
                        config.training.max_grad_norm,
                    )
                )
                if not torch.isfinite(torch.tensor(grad_norm)):
                    raise FloatingPointError(f"non-finite grad norm: {grad_norm}")
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                global_step += 1
                accum_position = 0
                step_time = time.perf_counter() - step_t0
                elapsed = time.perf_counter() - started
                epoch_fraction = samples_seen / float(n_samples)
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
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
                        gpu_memory_allocated_mb=allocated,
                        gpu_memory_reserved_mb=reserved,
                    )
                    csv_logger.append(row)
                    events.emit(
                        "step",
                        {
                            "global_step": global_step,
                            "loss": mean_loss,
                            "lr": lr,
                            "n_target": window_n_target,
                            "n_replay": window_n_replay,
                            "target_fraction": (
                                window_n_target / float(window_n_target + window_n_replay)
                                if (window_n_target + window_n_replay)
                                else 1.0
                            ),
                            "seen_fraction": (
                                window_n_replay / float(window_n_target + window_n_replay)
                                if (window_n_target + window_n_replay)
                                else 0.0
                            ),
                            **(
                                {
                                    "cum_target_fraction": mixer.cumulative_fractions()[0],
                                    "cum_seen_fraction": mixer.cumulative_fractions()[1],
                                }
                                if mixer is not None
                                else {}
                            ),
                        },
                    )
                    tb.log_train_step(
                        step=global_step,
                        loss=mean_loss,
                        learning_rate=lr,
                        grad_norm=grad_norm,
                        samples_per_second=sps,
                    )
                    _append_log(run_dir, f"step={global_step} loss={mean_loss:.8f} lr={lr:.8g}")
                if _should_save(global_step, config) or stop_requested["value"]:
                    _persist_checkpoint()
                if stop_requested["value"]:
                    break
            except Exception as error:
                if is_cuda_oom(error):
                    raise TrainError(
                        "CUDA OOM after auto-fit is forbidden; resolved "
                        f"physical_batch_size={physical} must fit for the whole run"
                    ) from error
                raise

        if (
            not stop_requested["value"]
            and global_step > 0
            and last_saved_step != global_step
        ):
            _persist_checkpoint()

        tb.close()
        if stop_requested["value"]:
            mark_interrupted(run_dir)
            return TrainResult(run_dir, global_step, "interrupted", final_checkpoint, last_loss)
        if global_step >= stop_at and final_checkpoint is not None:
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
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True, cancel_futures=True)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if lock.exists():
            lock.unlink()
