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
- No Linux GPU was available during M1 implementation. Revision status remains
  `resolved_m1_pending_hardware`; static doctor output cannot close M1.

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
  `collect_results` when pinned suite metadata is already present. Those
  commands still refuse training until later milestones.
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
- `create_libero_envs(..., n_envs=1)` always uses `episode_index=0`. The replay
  gate therefore uses the first demonstration of each task (`task_local_index=0`)
  so the public factory's init state matches the selected episode.
- Dataset `task_index` is not assumed equal to LIBERO env `task_id` on
  `libero_90` (73 unique texts vs 90 benchmark tasks). Target env IDs are pinned
  from the spec; seen env IDs are resolved by exact language match at runtime.
- Expert replay of the six gate trajectories still requires Linux `gpu` extra,
  EGL, action parquet (`--include-actions`), and simulator success. Static CI
  cannot close that hardware gate. M1 remains `resolved_m1_pending_hardware`.
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
  Auto-fit of physical batch `{4,2,1}` happens before the run directory exists.
  Weights are `weights.pt`; the static path still uses JSON toy weights.
  Frame decode stays in-process so the index cursor can resume exactly;
  `training.num_workers` is recorded but not used for DataLoader workers.
- Resume may override only `log_freq`, `destination`, `stop_after`,
  `backup_dir`, and `output_dir`. Dataset revision, split, trainable scope,
  optimizer, scheduler, batch, and seed are frozen.
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
- Failure videos are PPM frames (same encoding gate as expert replay). MP4/AV1
  waits on the M1 FFmpeg pin. Failure videos are never deleted by eval code.
- `eval_seen` probe tasks are frozen in
  `configs/splits/pseudo_target_splits.json` (three `libero_90` texts). Static
  smoke still defaults to `synthetic_seen` so it cannot look like `final_v1`.
- `--profile full` on Linux + CUDA runs live SmolVLA/LIBERO rollouts through
  the same JSONL resume loop. Target and language-control full eval stay
  refused until `configs/selected_seen_checkpoint.yaml` is frozen. No LeRobot
  eval CLI is called. `n_action_steps` is set to the eval
  `action_chunk_horizon` (10). Env task IDs are resolved by exact language
  match, not dataset `task_index`.
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
- `collect_results.py` drops `static_*` and `dev_soft_reset` rows. Cost-curve
  figures are SVG with x ticks `{0,5,10,25}` so the CPU extra does not need
  matplotlib. Spec PDF names remain a future optional export.
- `make_report_tables.py --bundle` checksums markdown/tables/figures only.
- TODO 23 **code** is the project SmolVLA trainer. The 100k GPU run itself
  still waits on a Linux CUDA VM. TODO 24 **code** is probe eval + selection;
  live probe rollouts wait on that same VM.

## Seen-pretrain trainer (TODO 23 code)

- Pinned `SmolVLAPolicy.forward(batch)` returns `(loss, loss_dict)` at
  `d451fe4…`. Action chunks use `delta_timestamps["action"] = range(chunk_size)/fps`
  with hub `chunk_size=50` and dataset fps 20.
- `LeRobotDataset(root=suite_dir, download_videos=False, revision=SHA, episodes=...)`
  is the pinned constructor. The local suite directory is
  `<datasets>/nvidia_LIBERO_LeRobot_v3/<SHA>/libero_90/`.
- Identity MEAN_STD stats remain smoke-only. Full training requires suite
  `stats.json` / `dataset.meta.stats`.
- After auto-fit, CUDA OOM is fatal (`physical_batch_size` is frozen).
- `make_pre_post_processors(..., preprocessor_overrides=device)` is attempted;
  a `TypeError` falls back to the signature already used by `smoke_inference.py`.
