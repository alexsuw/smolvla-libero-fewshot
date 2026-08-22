# Implementation notes

Здесь фиксируются проверенные расхождения между `PROJECT_SPEC.md` и API
pinned upstream revisions.

## M0

- CUDA, LeRobot, SmolVLA и LIBERO ещё не импортируются: это намеренно, потому
  что exact Linux runtime и upstream API проверяются в M1.
- `uv.lock` M0 покрывает только project-owned config/test toolchain. Он будет
  расширен и повторно проверен на Linux GPU VM в M1.
- Локальная машина разработки — macOS с Python 3.13. Project environment
  создаётся `uv` на Python 3.12, но это не считается доказательством CUDA/EGL
  совместимости.

## M1 implementation

- Pinned LeRobot commit `d451fe4f1f1b00a812f95aa9534389b5e42ab155`
  declares version `0.6.2` and Python `>=3.12`. Project exact patch is
  `3.12.8`, already exercised by macOS development and Ubuntu static CI.
- Runtime is a Linux-only `gpu` extra. `torch==2.11.0+cu128` and
  `torchvision==0.26.0+cu128` resolve only from the explicit cu128 index;
  normal CPU/macOS sync does not download CUDA wheels.
- LeRobot repository has LFS attributes although Python installation does not
  need binary assets. Bootstrap uses process-local Git filter overrides and
  never changes global Git config.
- `hf-libero==0.1.4` imports as `libero`. Its upstream metadata brings W&B
  transitively even without LeRobot's `training` extra. Project code never
  imports or logs to W&B and forces `WANDB_MODE=disabled` plus
  `WANDB_DISABLED=true`. Removing the transitive wheel would require a fork or
  an installation outside the `uv.lock` contract, so it remains installed but
  inactive.
- Pinned LeRobot maps the wrist camera to
  `observation.images.image2`, not `wrist_image`. Its
  `LiberoProcessorStep` also rotates both images by 180 degrees. M1 doctor
  verifies only executable two-camera/state shapes; M3 parity evidence decides
  the project-owned canonical mapping and transform.
- LIBERO imports can prompt when `~/.libero/config.yaml` is absent. Bootstrap
  downloads `lerobot/libero-assets` at pinned revision
  `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` and creates the config
  non-interactively; it refuses to overwrite a different existing config.
- FFmpeg `7.1.1` is a system pin, not equivalent to
  `imageio-ffmpeg==0.6.0` (Linux bundle 7.0.2). Bootstrap does not silently
  substitute it; full doctor requires exact 7.1.1 and a real AV1 round-trip.
- Full doctor on the first Linux GPU VM closed M1: revision status is
  `validated_m1`. Host needs exact FFmpeg 7.1.1 with libaom (Ubuntu 24.04 apt
  ships 6.1.1), NVIDIA EGL userspace (`libnvidia-gl` matching the driver), and
  membership in `video`/`render` for `/dev/dri`. The Linux `gpu` extra also
  installs a third-party site-packages `tests` package; repository
  `tests/__init__.py` makes local helpers win.

## M2 implementation

- Dataset files live under a revision-encoded directory
  `<datasets>/nvidia_LIBERO_LeRobot_v3/<40-char SHA>/`. Python never hard-codes
  `/mnt/vla` or `/content/drive`.
- Default download is metadata-only (`info.json`, `stats.json`, parquet). MP4
  files require `--include-videos` and exactly one `--suite`. Inspection sets
  `videos_decoded: false` and never decodes the corpus.
- `huggingface-hub==1.28.0` and `pyarrow==25.0.1` are the CPU `data` extra so
  macOS/CI can inspect metadata without the Linux `gpu` extra.
- Logical subsets write `subset_manifest.json` with nested `N=5/10/25` IDs and
  refuse to overwrite a different episode list. Videos are not copied.
- `assert_no_leakage` is invoked from `train_seen`, `train_target` and
  `collect_results` when pinned suite metadata is already present. `train_target`
  still refuses until the seen checkpoint YAML is frozen, then requires Linux CUDA.
- Wrist camera key in this dataset revision remains
  `observation.images.wrist_image`. Environment mapping to LeRobot `image2` is
  still an M3 decision.
- Pinned `tasks.parquet` stores task strings in `__index_level_0__` (pandas
  index leftover) plus integer `task_index`. Inspection accepts `task`,
  `task_text`, or `__index_level_0__`.

## M3 implementation

- Canonical policy wrist key is `observation.images.wrist_image`. Pinned
  LeRobot env/processor still emits `observation.images.image2`. The project
  adapter remaps that key once and refuses a dual alias.
- Image geometry: `LiberoProcessorStep` already flips H and W (rot180). The
  project adapter transform is frozen as `identity`. Enabling any second
  rotation/flip is a fatal configuration error.
- Gripper conversion `g_env = 1 - 2 g_dataset` happens only immediately before
  `env.step`. Dataset/training stay in `[0, 1]`. Binary runtime default:
  `< 0.5 -> +1`, `>= 0.5 -> -1`.
- `create_libero_envs(..., n_envs=1)` always constructs `episode_index=0`.
  Pinned LeRobot then walks `init_state_id` on every `reset`. Expert replay
  therefore pins `LiberoRuntime(init_state_id=...)` from the gate. NVIDIA
  dataset order is not `pruned_init` order: the first `libero_goal` wine
  episode (id 6) matches init state 1, not 0.
- Dataset `task_index` is not the LIBERO env `task_id`. Spec table 9/7/4 are
  parquet indices; benchmark ids for those texts are 0/1/2. Using the parquet
  index as `task_ids=[7]` loads `turn_on_the_stove` instead of bowl-on-stove.
  `resolve_env_task_id` matches exact language when LIBERO is importable;
  `configured` only disambiguates duplicate `libero_90` texts (book-in-caddy
  has env ids 73/78/81). CPU tests without LIBERO still return `configured`.
  The first dataset episode of that duplicated language (id 90) does not
  reproduce any of the 150 STUDY_SCENE1/2/3 init states; the gate uses episode
  139 with env id 73 (`STUDY_SCENE1`) init 0.
- Bootstrap sets `TOKENIZERS_PARALLELISM=false`. The env-value redactor must
  not treat that variable (or trivial `true`/`false`) as a secret; otherwise
  every JSON `false` in `rollouts.jsonl` becomes `[REDACTED]`.
- Expert replay of the six gate trajectories requires Linux `gpu` extra, EGL,
  action parquet (`--include-actions`), and simulator success. Static CI
  cannot close that hardware gate.
- `--save-video` writes PPM frames through the production replay path. AV1 MP4
  encoding stays on the exact FFmpeg 7.1.1 system pin from M1.

## M4 implementation

- Pinned hub checkpoint `lerobot/smolvla_base@c83c3163…` ships SO100 features
  (typically 3 cameras and 6D action). The loader overlays LIBERO
  `observation.images.image`, `observation.images.wrist_image`, 8D state and 7D
  action after load. Leftover hub camera keys are a fatal error.
- Architecture still uses `max_state_dim=32` and `max_action_dim=32`; the 7D
  LIBERO action is the unpadded prefix. The extra pretrained channels are not
  a substitute for LIBERO finetuning.
- LeRobot SmolVLAConfig flags are `freeze_vision_encoder`, `train_expert_only`
  (maps from project `freeze_vlm_backbone`), and `train_state_proj`. There is
  no upstream `train_action_expert` / `train_action_projections` flag; those
  are applied by the project allowlist after load.
- `assert_module_trainable_scope` must run before optimizer creation. Unknown
  or VLM parameters with `requires_grad=True` fail closed. Unused `lm_head`
  weights stay frozen.
- Static smoke (`--profile static`) never downloads weights. Full smoke needs
  Linux `gpu` extra and CUDA; identity MEAN_STD stats are smoke-only and must
  not be used for training.

## M5 implementation

- Pinned LeRobot `lerobot-train` (`src/lerobot/scripts/lerobot_train.py` at
  `d451fe4…`) uses `WandBLogger` and the `lerobot[training]` extra. The project
  trainer does not call that CLI. W&B stays disabled (`WANDB_MODE=disabled`).
- TensorBoard's Python package is in the Linux `gpu` extra. CPU/static logging
  always writes `tensorboard/tags.jsonl` with stable `train/*` tags; a
  `SummaryWriter` is used only when importable.
- Static `--profile static` proves atomic checkpoints, CSV/JSONL, manifests,
  registry rebuild, local dry-run-first backup, and exact 0→200 vs
  0→100→200 resume on a toy policy with SmolVLA-like parameter names.
  Weights are stored as IEEE-754 hex so JSON round-trips are bit-exact.
- `--profile full` on Linux + CUDA runs the project trainer (no `lerobot-train`).
  macOS and hosts without CUDA still fail closed with `no GPU training was started`.
  Auto-fit of physical batch `{32,16,8,4,2,1}` happens before the run directory exists.
  Weights are `weights.pt`; the static path still uses JSON toy weights.
  Frame decode stays in-process so the index cursor can resume exactly. For
  `gradient_accumulation=1`, a positive `training.num_workers` enables one
  ordered next-batch prefetch thread; the dataset backend already parallelizes
  the two camera streams. Prefetched indices are checkpointed, so resume does
  not skip them. Replay or accumulated runs keep synchronous loading.
  `save_torch_checkpoint` must import `file_checksums` / `sha256_file` /
  `verify_file_checksums`; a missing import fails the first GPU save at
  `every_steps` (this host: step 100, run left at
  `$VLA_RUNS_DIR/seen_smoke_200`). A static FFmpeg 7.1.1 executable is not
  enough for TorchCodec: `libav*.so` must be available on `LD_LIBRARY_PATH`.
  The checksummed `build_shared_ffmpeg.sh` host build removed the PyAV fallback.
  Outer multi-threaded `dataset.__getitem__` was rejected because it was slower
  with PyAV and unsafe with TorchCodec.
- Resume may override only `log_freq`, `destination`, `stop_after`,
  `backup_dir`, and `output_dir`. Dataset revision, split, trainable scope,
  optimizer, scheduler, batch, and seed are frozen. YAML `physical_batch_size:
  auto_fit` / `gradient_accumulation: auto` is not a contract change: resume
  loads the integers already frozen in the checkpoint and skips a second
  auto-fit so a crash cannot pick a different microbatch.
  Auto-fit tries divisors `{32,16,8,4,2,1}` largest-first; the RTX PRO 6000
  resolved `32/1`. Full `libero_90` throughput evidence is under
  `$VLA_DATA_ROOT/validation/M6/throughput`: 1.338 to 0.365 seconds/step and
  exact continuous-vs-resumed step-100 weight hashes.
- `sync_artifacts.py` default is dry-run and never deletes. Local directory
  destinations keep the M5 mirror. `file://` and `s3://` destinations use the
  object protocol: temporary prefix, size/checksum verify, remote
  `COMPLETED.json`, local `backup_status.json`. `s3:// --execute` needs boto3
  at runtime; dry-run does not. There is no `--delete`.
- `scripts/verify_backup.py` re-reads `COMPLETED.json` and checksums.

## Eval protocol implementation

- Evaluation is separate from training. `rollouts.jsonl` is append-only with
  the spec unique key; identical keys are skipped, conflicting outcomes fail
  closed. Incomplete runs resume by executing only missing keys.
- `--profile static` uses a toy env/policy so the protocol can be proven on
  CPU. Records are tagged `static_eval_v1` /
  `static_language_control_v1` / `static_seen_probe_v1` and must not enter
  final tables. Horizon and rollout count are shrunk only in that smoke
  overlay; tracked `final_v1` configs stay 20×300.
- Initial-state fingerprints exclude the instruction string so paired
  language-control rollouts can share a scene while traces diverge.
- Failure videos prefer FFmpeg `libaom-av1` MP4 via `shutil.which("ffmpeg")`
  (the VM `PATH` overlay, never a hard-coded `/mnt/vla` path). If AV1 encode
  fails, the writer falls back to PPM frames. Failures are never dropped.
- `eval_seen` probe tasks are frozen in
  `configs/splits/pseudo_target_splits.json` (three `libero_90` texts). Static
  smoke still defaults to `synthetic_seen` so it cannot look like `final_v1`.
- `--profile full` on Linux + CUDA runs live SmolVLA/LIBERO rollouts through
  the same JSONL resume loop. Target, zero-shot, and language-control full
  eval stay refused until `configs/selected_seen_checkpoint.yaml` is frozen.
  Zero-shot (`scripts/eval_zero_shot.py`) uses that frozen hash, `n_demos=0`,
  and an empty training episode list. Language control
  (`scripts/eval_language_control.py`) uses the same origin and seeds: paired
  correct/wrong instructions, matching initial-state fingerprints, and matching
  checkpoint hashes. `--run-dir` is refused for both so they cannot score every
  seen-pretrain step. No LeRobot eval CLI is called.
  `n_action_steps` is set to the eval `action_chunk_horizon` (10). Env task
  IDs are resolved by exact language match, not dataset `task_index`.
  Live eval must apply the LeRobot action postprocessor (MEAN_STD unnormalize
  + CPU move) to `select_action` output before `dataset_action_to_env`.
  Dropping that pipeline leaves gripper in z-score space, which fails the
  dataset `[0, 1]` gate. After unnormalize, gripper is clipped to `[0, 1]`
  because pinned UnnormalizerProcessorStep does not clip; env-space `-1`
  without unnormalize is still refused.
- Seen checkpoint selection scores only the three probe slugs, drops NaN/
  unstable rows, rejects `static_*` protocols and any target-task slug, then
  takes the earliest checkpoint within `tolerance_success=0.02` of the best
  mean. If every stable score is inside that band, the fallback is step
  100000. `select_seen_checkpoint.py` is dry-run until `--write`.


## Object storage, predictions, calibration, reporting

- Predictions are in `predictions.md` and must not be rewritten after the
  real target grid starts.
- Pseudo-target tasks are three `libero_90` texts from the replay gate, not
  the held-out `libero_goal` targets. Train YAML values are checked against
  `configs/calibration.yaml`. The selected seen-checkpoint hash stays unset
  until `select_seen_checkpoint.py --write` on the VM.
- `collect_results.py` drops `static_*` and `dev_soft_reset` rows. Language
  control reports both `final_language_control_v1` (live) and the legacy
  `language_control_v1` alias. Cost-curve figures are SVG with x ticks
  `{0,5,10,25}` so the CPU extra does not need matplotlib. Spec PDF names
  remain a future optional export.
- `make_report_tables.py --bundle` checksums markdown/tables/figures only.
- TODO 23 **code** is the project SmolVLA trainer. The 100k GPU run is the
  live `seen_expert` job after the 200-step smoke. TODO 24 **code** is probe eval + selection;
  live probe rollouts wait on that same VM. TODO 25 (seen LoRA) is skipped.
  TODO 26 **code** is `eval_zero_shot.py` (3 tasks × ≥20, empty train list,
  frozen seen hash).   TODO 27 **code** is `eval_language_control.py` (paired
  correct/wrong, same seeds/states/hash). TODO 28 **code** is `train_target.py`
  baseline (no LoRA/replay). TODO 29 **code** is `eval_target.py --run-dir`
  plus `verify_baseline_eval.py` (≥20 rollouts and failure videos per complete
  checkpoint). TODO 30 **code** is `train_target.py --config target_lora.yaml`
  (PEFT wrap, merged-free adapter sidecar, same nested episodes). TODO 31
  **code** is Replay-LoRA: 75/25 mixer over `libero_90` only, same LoRA wrap.
  GPU runs wait on the freeze.

## Seen-pretrain trainer (TODO 23 code)

- Pinned `SmolVLAPolicy.forward(batch)` returns `(loss, loss_dict)` at
  `d451fe4…`. Action chunks use `delta_timestamps["action"] = range(chunk_size)/fps`
  with hub `chunk_size=50` and dataset fps 20.
- `LeRobotDataset(root=suite_dir, download_videos=False, revision=SHA, episodes=...)`
  is the pinned constructor. The local suite directory is
  `<datasets>/nvidia_LIBERO_LeRobot_v3/<SHA>/libero_90/`.
- Identity MEAN_STD stats remain smoke-only. Full seen training uses suite
  `stats.json` / `dataset.meta.stats`. Target baseline overlays subset-local
  MEAN_STD for `observation.state` and `action` from the selected episodes;
  image stats stay IDENTITY.
- After auto-fit, CUDA OOM is fatal (`physical_batch_size` is frozen).
- `make_pre_post_processors(..., preprocessor_overrides=device)` is attempted;
  a `TypeError` falls back to the signature already used by `smoke_inference.py`.

## Target baseline trainer (TODO 28 code)

- `train_target.py` loads the frozen seen `weights.pt` only (fresh AdamW).
  Baseline YAML forbids LoRA/replay. Episode IDs are nested prefixes from
  `configs/splits/target_splits.json`. Stop is `min(100 epochs, 12000 steps)`
  with `sample_with_replacement: true`. The final step is always checkpointed
  even when it is not on `save_steps`.
- `--print-grid` lists the 18 independent cells. `eval_target.py --run-dir`
  evaluates every complete baseline checkpoint. `verify_baseline_eval.py` is
  the complete-cell checker: ≥20 `final_v1` rollouts, traces, failure videos,
  nested episode IDs. Static rows cannot close the grid.
- Epoch length is `ceil(n_samples / effective_batch_size)` optimizer steps,
  not physical micro-batches.

## Target LoRA ablation (TODO 30 code)

- Same frozen seen origin, same nested `libero_goal` episode IDs, same
  `min(100 epochs, 12000 steps)` cap, no replay. Config:
  `configs/train/target_lora.yaml` (r=64, alpha=64, LR `1e-3`).
- Wrap uses `peft.LoraConfig` + `get_peft_model` after origin `weights.pt`
  load and **before** the allowlist/optimizer. We do not call `lerobot-train`.
- Target modules are the pinned expert `q_proj`/`v_proj` regex
  `model\\.vlm_with_expert\\.lm_expert\\..*\\.(q|v)_proj`. This is the expert half of
  LeRobot `SmolVLAPolicy._get_default_peft_targets`. State/action projections
  stay full-rank because `target_lora.yaml` sets `train_state_projection` /
  `train_action_projections` true and `train_action_expert` false.
- Checkpoints write `adapter/adapter_config.json` + `adapter_model.pt` with
  `merged: false`. Eval wraps again and loads adapters; `merge=True` /
  `merge_and_unload` is refused.
- `--print-grid --config configs/train/target_lora.yaml` lists the 18 LoRA
  cells. Eval: `--train-config configs/train/target_lora.yaml --print-grid`.
- Replay-LoRA uses `--config configs/train/target_replay_lora.yaml`.
- Darwin / unfrozen seen YAML fail closed with `no GPU training was started`.

## Replay-LoRA (TODO 31 code)

- Same frozen seen origin, nested target episode IDs, LoRA wrap, and target
  subset MEAN_STD as the LoRA ablation. Replay frames are normalized with
  those same target stats so the ablation is not confounded by a second scaler.
- Mixer is deterministic from the train seed: 75% target / 25% `libero_90`
  via remainder carry so the long-run fraction matches. `libero_goal` pool or
  task text fails closed. Seen frames do not change N or epoch length.
- Each optimizer-step event logs `n_target` / `n_replay` and cumulative
  fractions. Replay cursor is independent (`seed+1`) and always samples with
  replacement from the full `libero_90` frame pool (all 73 tasks).
- Eval: `--train-config configs/train/target_replay_lora.yaml`.
