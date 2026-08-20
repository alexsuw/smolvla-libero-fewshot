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
