# Project status

## M0 — Repository skeleton and contracts

Status: complete locally; Ubuntu verification is enforced by
`.github/workflows/m0-contracts.yml` on every push and pull request.

Completed:

- Python package, repository structure, config templates and thin notebooks;
- strict Pydantic YAML contracts with unknown-key rejection;
- exact tracked target prefixes and final evaluation seeds;
- 27 safe CLI `--help` paths; compute commands fail before allocation;
- environment-only runtime paths;
- Git guard for secrets, files over 10 MB and runtime payloads;
- frozen lightweight M0 environment on Python 3.12.

Acceptance commands:

```bash
uv sync --frozen
make check
```

Local result on 2026-08-21:

- `uv sync --frozen`: passed with Python 3.12.8;
- strict config/split/seed validation: passed;
- CLI help validation: 27 commands passed;
- Git safety: passed;
- pytest: 41 passed.

Local evidence:

```text
artifacts/validation/M0/acceptance.log
artifacts/validation/M0/uv_sync.log
artifacts/validation/M0/environment.txt
```

No dataset download, model load, simulator run, training, GPU allocation or
external artifact upload was performed in M0.

## M1 — Pinned runtime and doctor

Status: complete. Full doctor passed on the Linux GPU VM; revision state is
`validated_m1`.

Completed:

- universal lock with Linux-only cu128/LeRobot/LIBERO/PEFT runtime;
- exact Python 3.12.8 and upstream model/dataset/source/assets revisions;
- secure local and remote revision validation;
- VM/Colab bootstrap with pinned LIBERO assets and non-overwriting manifests;
- environment-only paths, Google Drive durability detection and round-trip;
- static/full doctor profiles with structured JSON/Markdown reports;
- full-profile checks for driver/CUDA/BF16, EGL, two cameras, 8D state,
  FFmpeg 7.1.1 AV1, disk reserve and durable storage;
- Colab Drive launcher and final RTX VM handoff documentation.

Static result on 2026-08-21:

- `uv lock --check`: passed, 165 packages resolved;
- remote model/dataset/source/assets revisions: passed;
- CLI help validation: 28 commands passed;
- Git safety: passed;
- pytest: 54 passed, 2 hardware/storage tests skipped intentionally;
- static doctor: required checks passed, `acceptance_complete=false` by design.

Local evidence:

```text
artifacts/validation/M1/acceptance-static.log
artifacts/validation/M1/upstream_revisions.json
artifacts/validation/M1/upstream_revisions.log
artifacts/validation/M1/environment.txt
artifacts/validation/M1/doctor-static-*/
```

Hardware result on 2026-08-21 (Linux GPU VM):

- `uv sync --frozen --extra gpu`: passed, Python 3.12.8, `torch==2.11.0+cu128`;
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver `580.173.02`,
  CUDA 12.8, `sm_120`, BF16 supported, ~97 GB VRAM;
- MuJoCo EGL render 32×32×3; LIBERO `libero_goal` two-camera 256² + 8D state;
- system FFmpeg `7.1.1` with libaom AV1 round-trip;
- durable storage `/mnt/vla` (not ephemeral), >50 GB reserve;
- `VLA_RUN_GPU_TESTS=1 VLA_RUN_STORAGE_TESTS=1 pytest -m "gpu or integration"`:
  2 passed, 1 skipped (live HF metadata).

Persistent evidence:

```text
/mnt/vla/bootstrap/20260821T233035Z/
/mnt/vla/doctor/20260821T233435Z/   # required checks passed, status still pending
/mnt/vla/doctor/20260821T233720Z/   # acceptance_complete=true after validated_m1
```

Host setup that was missing on the stock CUDA image: FFmpeg 7.1.1+libaom,
`libnvidia-gl-580` EGL ICD, and `video`/`render` group access to `/dev/dri`.

## M2 — Dataset inspection and exact split

Status: implementation complete on CPU. Unit contracts do not require a GPU.
Live Hugging Face metadata download is optional evidence and is not a paid
hardware gate. M1 is `validated_m1`; M2 itself does not require a GPU.

Completed:

- metadata-only pinned download with resume and revision-encoded paths;
- inspection of schema, counts, task texts, episode IDs and stats without
  decoding videos;
- exact first-25 target prefixes and nested `N=5/10/25` budgets;
- fail-closed no-leakage gate wired into `train_seen`, `train_target` and
  `collect_results`;
- logical subset manifests that refuse a different overwrite.

Acceptance commands:

```bash
uv sync --frozen --extra data
make check-m2
```

Local result on 2026-08-21:

- `uv lock --check`: passed, 165 packages, extras `data` and `gpu`;
- CLI help validation: 28 commands passed;
- Git safety: passed;
- pytest: 67 passed, 3 skipped without GPU/storage/live-HF env vars;
- live metadata-only download: 19 files, 5.2 MB, no MP4;
- `inspect_dataset`: `acceptance_complete=true`, `videos_decoded=false`;
- `verify_split` and `verify_no_leakage`: passed on pinned revision
  `e5907374380b8f96511957e6ba5582be52a1e179`.

Local evidence:

```text
artifacts/validation/M2/inspection.json
artifacts/validation/M2/inspection.md
artifacts/validation/M2/target_episode_ids.json
artifacts/validation/M2/verify-split.json
artifacts/validation/M2/verify-leakage.json
artifacts/validation/M2/acceptance.log
```

No training, simulator replay, model load or GPU allocation is part of M2.

## M3 — Environment adapters and expert replay

Status: complete on this host. The six-episode expert-replay gate succeeded
with simulator `success=1` after mapping LIBERO env ids by language (not
dataset `task_index`) and pinning `pruned_init` rows.

Completed:

- explicit camera map `image2` → `observation.images.wrist_image`;
- frozen orientation contract: LeRobot rot180 only, project transform identity;
- 8D state flatten matching pinned `LiberoProcessorStep`;
- gripper/action postprocessor with dual-space traces;
- LIBERO wrapper over pinned `create_libero_envs` (hard reset, relative, 256²);
- observation parity CLI and six-episode replay gate;
- train configs now record `env.control_mode: relative` and `action.dim: 7`.

Acceptance commands:

```bash
uv sync --frozen --extra data
make check-m3
```

Local result on 2026-08-21:

- `uv sync --frozen --extra data` plus `make check-m3`: passed;
- CLI help validation: 28 commands passed;
- Git safety: passed;
- pytest: 90 passed, 3 skipped without GPU/storage/live-HF env vars;
- endpoint gripper tests: `0→+1`, `1→-1`, `0.5→0`;
- double-flip and double gripper conversion rejected;
- replay gate: 3 target first-demos + 3 diverse seen first-demos;
- parity bundle writes candidate transforms without decoding videos.

Local evidence:

```text
artifacts/validation/M3/parity/
artifacts/validation/M3/acceptance.log
```

Hardware result on 2026-08-22 (Linux GPU VM):

- action parquet downloaded (`--include-actions`, 30 files, no MP4);
- `check_observation_parity.py --with-env` wrote `/mnt/vla/validation/M3/parity-env`;
- `replay_expert.py --all-gate --save-video` succeeded for all six gate
  episodes (3 `libero_goal` + 3 `libero_90`), traces and PPM frames saved.

Persistent evidence:

```text
/mnt/vla/datasets/nvidia_LIBERO_LeRobot_v3/e5907374380b8f96511957e6ba5582be52a1e179/
/mnt/vla/validation/M2/inspect
/mnt/vla/validation/M3/parity-env
/mnt/vla/validation/M3/replay
```

## M4 — SmolVLA inference and trainable scope

Status: complete on this host. Full SmolVLA smoke loaded pinned weights on
CUDA and `env.step` accepted the converted 7D action. M1 is `validated_m1`.

Completed:

- pinned SmolVLA loader with explicit LIBERO 2-camera / 8D / 7D feature overlay;
- fail-closed trainable allowlist mapped onto LeRobot freeze flags;
- `trainable_parameters.txt` writer used before any optimizer creation;
- static smoke CLI that never downloads weights.

Acceptance commands:

```bash
uv sync --frozen --extra data
make check-m4
```

Local result on 2026-08-21:

- `make check-m4`: passed on CPU;
- CLI help validation: 28 commands passed;
- Git safety: passed;
- pytest: 102 passed, 4 skipped without GPU/storage/live-HF env vars;
- LIBERO feature contract rejects hub SO100 cameras and 6D actions;
- seen-pretrain allowlist trains action expert + state/action projections only;
- unintended VLM `requires_grad` fails closed;
- static smoke `acceptance_complete=false` by design.

Local evidence:

```text
artifacts/validation/M4/smoke_inference.json
artifacts/validation/M4/smoke_inference.md
artifacts/validation/M4/acceptance.log
```

Hardware result on 2026-08-22 (Linux GPU VM):

- `smoke_inference.py --profile full --with-env`: `acceptance_complete=true`;
- device `cuda`; hub SO100 features overlaid with LIBERO 2-camera / 8D / 7D;
- `env_step.accepted=true` on `libero_goal` bowl-on-stove (dummy action
  `is_success=false` as expected).

Persistent evidence:

```text
/mnt/vla/validation/M4/full/smoke_inference.json
/mnt/vla/validation/M4/full/trainable_parameters.txt
/mnt/vla/logs/m4_smoke_inference.log
```

## M5 — Training/checkpoint/resume smoke

Status: complete on this host. Live SmolVLA 200-step training wrote atomic
torch checkpoints on durable storage. M1 is `validated_m1`; M3/M4 hardware
gates are closed.

Completed:

- TensorBoard JSONL fallback + CSV/JSONL/plain-text logs; W&B stays off;
- immutable `manifest.json`, registry rebuild from manifests;
- atomic checkpoint directories with `COMPLETED.json`, checksums, and
  fresh-instance load;
- exact 0→200 vs 0→100→200 comparison, including a fresh subprocess resume;
- local dry-run-first backup; prune remains inventory-only.

Acceptance commands:

```bash
uv sync --frozen --extra data
make check-m5
```

Local result on 2026-08-21:

- `make check-m5`: passed on CPU;
- CLI help validation: 28 commands passed;
- Git safety: passed;
- pytest: 118 passed, 4 skipped without GPU/storage/live-HF env vars;
- exact 0→200 vs 0→100→200 comparison passed with bit-exact weights;
- `python scripts/train_seen.py` without `--profile static` fails before GPU
  allocation;
- static `acceptance_complete=true` for the toy resume protocol only.

Local evidence:

```text
artifacts/validation/M5/resume_compare.json
artifacts/validation/M5/resume_compare.md
artifacts/validation/M5/registry.csv
```

Hardware result on 2026-08-22 (Linux GPU VM):

- `libero_90` and `libero_goal` videos on the pinned revision root;
- first 200-step attempt `/mnt/vla/runs/seen_smoke_200` failed at step 100
  (`NameError: file_checksums`); that directory is kept;
- retry `train_seen.py --profile full` →
  `/mnt/vla/runs/seen_smoke_200_r2` `status=completed` at step 200 with
  `step_000100` and `step_000200` `COMPLETED.json` checkpoints, bf16.

Persistent evidence:

```text
/mnt/vla/runs/seen_smoke_200_r2/
/mnt/vla/runs/seen_smoke_200_r2.console.log
/mnt/vla/logs/seen_smoke_200.tmux.log
/mnt/vla/runs/seen_smoke_200/   # failed first attempt, not deleted
```

## Eval protocol — fixed-seed rollouts (TODO 19)

Status: implementation and CPU/static acceptance complete; live LIBERO/SmolVLA
rollouts remain deferred until Linux CUDA/`gpu` extra is available and
M1/M3/M4 hardware gates pass.

Completed:

- unique rollout key, JSONL resume, and conflicting-duplicate refusal;
- Wilson 95% CI helper;
- traces for every rollout; failure videos always, first success per cell;
- paired language-control fingerprints and action divergence;
- `eval_target` / `eval_seen` / `eval_language_control` with `--profile static`.

Acceptance commands:

```bash
uv sync --frozen --extra data
make check-eval-protocol
```

Local result on 2026-08-21:

- `make check-eval-protocol`: passed on CPU;
- CLI help validation: 28 commands passed;
- Git safety: passed;
- pytest: 129 passed, 4 skipped without GPU/storage/live-HF env vars;
- `--profile full` fails before GPU allocation.

Local evidence:

```text
artifacts/validation/eval-protocol/target/
artifacts/validation/eval-protocol/language/
artifacts/validation/eval-protocol/seen/
```

Deferred hardware gates:

- live LIBERO hard-reset rollouts with pinned SmolVLA;
- AV1 failure videos through the M1 FFmpeg 7.1.1 pin;
- 20-rollout `final_v1` cells.

## Object storage, predictions, calibration, reporting (TODO 20–22, 32–35 CPU)

Status: software and CPU/static acceptance complete. Live S3/`boto3` round-trip
and paid GPU runs remain deferred.

Completed:

- dry-run-first object sync for `file://` and `s3://` with checksum verify,
  `COMPLETED.json`, local `backup_status.json`, and no delete;
- committed `predictions.md` before any target results;
- frozen pseudo-target tasks inside `libero_90` and matching train YAMLs;
- `collect_results` / cost-curve SVG / report tables / checksummed bundle;
- three named failure-analysis slots without numerical claims.

Acceptance commands:

```bash
uv sync --frozen --extra data
make check-reporting
```

Local result on 2026-08-21:

- pytest: 143 passed, 4 skipped;
- CLI help validation: 29 commands passed;
- Git safety: passed;
- `file://` object sync execute + `verify_backup`: verified, deleted=0;
- `collect_results` drops `static_*` rows; cost-curve SVG keeps x ticks 0/5/10/25.

Local evidence:

```text
artifacts/validation/object-sync/
artifacts/validation/reporting/
```

Deferred hardware / paid runs (TODO 23 GPU, 24 live probes, 25–31, 36 live S3):

- optimized 100k `libero_90` seen-pretrain and seen probes after it finishes;
- zero-shot, language control, baseline grid, LoRA on real targets;
- verified remote backup of the final bundle to a real bucket.

## Seen-pretrain trainer (TODO 23 code)

Status: GPU 100k run complete on this host. The original `physical=4` run was
cleanly interrupted at step 416 and preserved. The optimized run finished at
step 100000. `configs/selected_seen_checkpoint.yaml` is now `frozen`
at that step after 10-seed `libero_90` probes.

Completed:

- project-owned SmolVLA loop (`train_seen.py --profile full`) that never calls
  `lerobot-train` / WandBLogger;
- allowlist before AdamW; auto-fit `{32,16,8,4,2,1}` before the run directory
  exists;
- torch checkpoints (`weights.pt`) with checksums and `COMPLETED.json`;
- deterministic frame cursor for resume; suite stats for MEAN_STD;
- shared pinned FFmpeg 7.1.1 enables TorchCodec; one ordered next-batch
  prefetch overlaps decode with CUDA and checkpoints pending indices;
- Darwin / no-CUDA still fail closed with `no GPU training was started`.

RTX PRO 6000 throughput acceptance on 2026-08-22:

- original `physical=4`, accumulation 8, PyAV: `1.338 s/step` → 37.18 h;
- `physical=32`, accumulation 1, TorchCodec + ordered prefetch:
  `0.365 s/step` → 10.15 h for the unchanged 100k schedule;
- full 3,921 episodes / 569,249 frames / 73 tasks retained; no target data;
- SIGTERM at step 51 resumed to step 100 with the same weight SHA-256 as the
  uninterrupted run:
  `8d61568909ec36550a4f525aef0720517285c429656970ddf18eaaa0c954d8f1`;
- full test suite: 210 passed, 8 skipped.

Persistent evidence:

```text
/mnt/vla/validation/M6/throughput/summary.md
/mnt/vla/runs/seen__expert__libero90__nall__s42__20260822T010019Z__gd4b8fb8/
/mnt/vla/runs/seen_expert_100k/checkpoints/step_000416/
```

Completed 100k run (2026-08-22):

- output: `/mnt/vla/runs/seen__expert__libero90__nall__s42__20260822T010019Z__gd4b8fb8`
- `status=completed`, `global_step=100000`, `samples_seen=3200000`
- final loss `0.23494`; weights SHA-256
  `2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88`
- 22 complete checkpoints (20 scheduled 5k–100k plus SIGTERM 1739/1833)
- no target data; TensorBoard/CSV/JSONL written on `/mnt/vla`

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_full_train_cpu.py tests/smoke/test_cli_help.py
uv run python scripts/train_seen.py --help
```

## Seen probes and checkpoint freeze (TODO 24 code)

Status: complete on this VM. Tracked YAML is `frozen` at step 100000.

Completed:

- Live `libero_90` probes only (never `libero_goal`). Stage 1: 20k/40k/60k/80k/100k
  × 5 seeds. Stage 2: 60k and 80k to 10 seeds, then 100k also to 10 seeds
  after a 5-seed mean tie with 80k.
- 10-seed means: 60k 0.700 (21/30), 80k 0.767 (23/30), 100k 0.800 (24/30).
  Earliest within 0.02 of best is only 100k (`used_fallback=false`).
- Frozen hash
  `2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88`
  at `checkpoints/step_100000` under run
  `seen__expert__libero90__nall__s42__20260822T010019Z__gd4b8fb8`.
- Interrupt/5k leftover cells kept on disk and ignored.
- `select_seen_checkpoint.py --write` is idempotent; static rows cannot freeze.

Persistent evidence:

```text
/mnt/vla/eval/seen_probes__gd4b8fb8/report/summary.md
/mnt/vla/eval/seen_probes__gd4b8fb8/ten_seed_selection.json
/mnt/vla/validation/TODO24/
```

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_seen_selection.py tests/integration/test_eval_resume.py
uv run python scripts/eval_seen.py --help
uv run python scripts/select_seen_checkpoint.py --help
```

## Zero-shot final eval (TODO 26)

Status: complete on this VM. Official protocol is `zero_shot_v2_seen_stats`
(frozen step 100000, 3 `libero_goal` tasks × 20, empty train list, hash
`2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88`,
normalization suite `libero_90`).

Result: **1/20** `drawer_middle` (seed 1001), **0/20** `bowl_stove`,
**0/20** `wine_cabinet` (overall **1/60**). The earlier `final_v1` 0/60
used held-out `libero_goal` MEAN/STD and is protocol-invalid; those
artifacts stay on disk. This is N=0 on the cost curve, not a reason to
unfreeze or retune on target success.

Persistent evidence:

```text
/mnt/vla/eval/zero_shot_v2_seen_stats/report/summary.md
/mnt/vla/eval/zero_shot_v2_seen_stats/report/results_long.csv
/mnt/vla/validation/TODO26_v2/
/mnt/vla/eval/zero_shot/          # contaminated final_v1 0/60, kept
/mnt/vla/validation/TODO26/
```

Acceptance commands:

```bash
uv run python scripts/eval_zero_shot.py --help
uv run python scripts/export_zero_shot_report.py --help
```

## Target baseline (TODO 28 code)

Status: software complete on CPU. Seen YAML is `frozen`. GPU 18-cell runs
wait until the persistent volume is large enough. Seen LoRA (TODO 25) is
skipped and does not delay this path.

Completed:

- `train_target.py` continues from the frozen seen `weights.pt` with Action
  Expert + projections only (no LoRA, no replay);
- nested `5/10/25` episode prefixes, `sample_with_replacement`,
  `min(100 epochs, 12000 steps)`, final checkpoint always saved;
- `--print-grid` lists the 18 independent cells;
- `eval_target.py --run-dir` evaluates every complete baseline checkpoint;
- Darwin / unfrozen YAML / Replay-LoRA configs fail closed with
  `no GPU training was started`. Target LoRA uses `--config target_lora.yaml`.

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_target_baseline.py
uv run python scripts/train_target.py --help
uv run python scripts/train_target.py --print-grid
```

## Zero-shot final eval (TODO 26 code)

Status: software complete on CPU. Live 3×20 rollouts wait until the seen
checkpoint YAML is `frozen`.

Completed:

- `eval_zero_shot.py` runs all three `libero_goal` targets when `--task` is
  omitted; `--print-grid` lists per-task commands;
- `n_demos=0`, empty `training_episode_ids`, method `seen`;
- full profile loads the frozen seen URI and checks `sha256`; `--run-dir` is
  refused;
- Darwin / unfrozen YAML fail closed with `no GPU evaluation was started`.

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_zero_shot.py tests/integration/test_eval_resume.py
uv run python scripts/eval_zero_shot.py --help
uv run python scripts/eval_zero_shot.py --print-grid
```

## Language control (TODO 27)

Status: complete on this VM. Protocol `final_language_control_v1`, frozen
step 100000, `libero_90` stats, 3 tasks × 20 seeds × correct/wrong.

Result: correct **1/60** (same `drawer_middle` seed 1001 as zero-shot),
wrong **0/60**. All 60 pairs share fingerprints and the frozen hash.
Correct cells reproduce zero-shot v2 fingerprints and successes 60/60.
Mean action L2: drawer 0.582, bowl 0.998, wine 0.966. Seed 1001 is
correct-success / wrong-fail (L2 0.477). Low wrong-success is not the
proof; the trajectories diverge. Do not retune on these numbers.

Persistent evidence:

```text
/mnt/vla/eval/language_control/report/summary.md
/mnt/vla/eval/language_control/report/pairs.csv
/mnt/vla/validation/TODO27/
```

Acceptance commands:

```bash
uv run python scripts/eval_language_control.py --help
uv run python scripts/export_language_control_report.py --help
uv run python scripts/watch_eval_progress.py --help
```

## Baseline eval checker (TODO 29 code)

Status: software complete on CPU. Live ≥20 rollouts per checkpoint wait on the
VM after freeze and the 18 training runs.

Completed:

- `eval_target.py --print-grid` lists 18 `--run-dir` eval commands;
- `verify_baseline_eval.py` requires every complete checkpoint to have ≥20
  `final_v1` rollouts, traces, and a video for every failure;
- nested episode IDs and train seed/task must match; static rows cannot close
  the cell.

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_baseline_eval.py
uv run python scripts/eval_target.py --print-grid
uv run python scripts/verify_baseline_eval.py --help
```

## Target LoRA ablation (TODO 30 code)

Status: software complete on CPU. Live GPU cells wait on the VM after freeze
and the 18-cell baseline.

Completed:

- `train_target.py --config configs/train/target_lora.yaml` wraps PEFT LoRA
  after the frozen seen `weights.pt` load and before the allowlist/optimizer;
- same nested episode IDs, seeds, and step cap as baseline; no replay;
- adapter sidecar is merged-free; eval loads with `--train-config`.

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_target_lora.py tests/unit/test_freezing.py tests/unit/test_target_baseline.py
uv run python scripts/train_target.py --config configs/train/target_lora.yaml --print-grid
uv run python scripts/eval_target.py --train-config configs/train/target_lora.yaml --print-grid
```

## Replay-LoRA (TODO 31 code)

Status: software complete on CPU. Live GPU cells wait on the VM after freeze,
baseline, and the LoRA ablation.

Completed:

- `train_target.py --config configs/train/target_replay_lora.yaml` mixes 75%
  nested target frames with 25% `libero_90` (never `libero_goal`);
- same LoRA wrap, origin, episode IDs, and target-subset MEAN_STD as LoRA;
- mixer is seeded and resumable; step events log per-window fractions;
- seen frames do not change N or the epoch cap.

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_replay_mixer.py tests/unit/test_target_lora.py
uv run python scripts/train_target.py --config configs/train/target_replay_lora.yaml --print-grid
uv run python scripts/eval_target.py --train-config configs/train/target_replay_lora.yaml --print-grid
```

## Next milestone

M7 live eval is done (zero-shot 1/60, language control 1/60 vs 0/60
with diverging paired trajectories). Next is TODO 28, the 18-cell
baseline, only after the persistent volume is actually ~512 GB. Do not
retune on target success. Seen LoRA is skipped.
