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

Status: implementation and static validation complete; full hardware acceptance
is consciously deferred at the user's request until the final GPU purchase.
Revision state remains `resolved_m1_pending_hardware`, not `validated_m1`.

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

Deferred hardware gates:

- install `gpu` extra on Linux;
- verify NVIDIA driver `>=570.86.10`, CUDA architecture and BF16;
- render MuJoCo/LIBERO through EGL;
- reset/step with two cameras and 8D state;
- verify exact system FFmpeg 7.1.1 AV1;
- verify persistent SSD or mounted Google Drive.

These checks must pass before revision status changes to `validated_m1` and
before any training. Static CI can never close this gate.

## M2 — Dataset inspection and exact split

Status: implementation complete on CPU. Unit contracts do not require a GPU.
Live Hugging Face metadata download is optional evidence and is not a paid
hardware gate. M1 revision status stays `resolved_m1_pending_hardware`.

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

M1 revision status is unchanged: `resolved_m1_pending_hardware`. No training,
simulator replay, model load or GPU allocation is part of M2.

## M3 — Environment adapters and expert replay

Status: implementation and CPU/unit acceptance complete; live simulator replay
of the six gate trajectories is deferred until Linux EGL/`gpu` extra is
available. M1 revision status stays `resolved_m1_pending_hardware`.

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

Deferred hardware gates:

- `uv sync --frozen --extra gpu` on Linux;
- `python scripts/check_observation_parity.py --with-env`;
- download action parquet (`--include-actions`) then
  `python scripts/replay_expert.py --all-gate --save-video`.

These must succeed with simulator `success=1` and saved traces/videos before
training. Static CI cannot close this gate.

## M4 — SmolVLA inference and trainable scope

Status: implementation and CPU/static acceptance complete; live weight load and
env-accepted action remain deferred until Linux CUDA/`gpu` extra is available.
M1 revision status stays `resolved_m1_pending_hardware`.

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

Deferred hardware gates:

- `uv sync --frozen --extra gpu` on Linux with CUDA;
- `python scripts/smoke_inference.py --config configs/train/smoke.yaml --profile full`;
- optional `--with-env` to prove `env.step` accepts the converted 7D action.

These must pass before M5 training. Static CI cannot close this gate.

## M5 — Training/checkpoint/resume smoke

Status: implementation and CPU/static acceptance complete; live SmolVLA
200-step training remains deferred until Linux CUDA/`gpu` extra is available
and M1/M3/M4 hardware gates pass. M1 revision status stays
`resolved_m1_pending_hardware`.

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

Deferred hardware gates:

- `uv sync --frozen --extra gpu` on Linux with CUDA;
- `python scripts/train_seen.py --config configs/train/smoke.yaml --profile full`;
- Colab 0→100→200 on pinned SmolVLA with dataset videos.

These must pass before paid long seen-pretrain (M6). Static CI cannot close
the SmolVLA training gate.

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

- 100k `libero_90` seen-pretrain and seen probes;
- zero-shot, language control, baseline grid, LoRA on real targets;
- verified remote backup of the final bundle to a real bucket.

## Seen-pretrain trainer (TODO 23 code)

Status: software complete on CPU; the 100k CUDA run is deferred to the VM.

Completed:

- project-owned SmolVLA loop (`train_seen.py --profile full`) that never calls
  `lerobot-train` / WandBLogger;
- allowlist before AdamW; auto-fit `{4,2,1}` before the run directory exists;
- torch checkpoints (`weights.pt`) with checksums and `COMPLETED.json`;
- deterministic frame cursor for resume; suite stats for MEAN_STD;
- Darwin / no-CUDA still fail closed with `no GPU training was started`.

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_full_train_cpu.py tests/smoke/test_cli_help.py
uv run python scripts/train_seen.py --help
```

## Seen probes and checkpoint freeze (TODO 24 code)

Status: software complete on CPU; live probe rollouts and the frozen hash wait
on the VM. Tracked `configs/selected_seen_checkpoint.yaml` stays
`pending_seen_pretrain`.

Completed:

- `eval_seen.py --profile full` runs LIBERO/SmolVLA probes (`--run-dir` covers
  every complete checkpoint × three frozen slugs);
- target/language full eval refuses until the seen checkpoint is frozen;
- `select_seen_checkpoint.py` dry-run-first: earliest within 0.02 of best,
  else step 100000; never reads target success; `--write` is idempotent;
- static `*_v1` probe rows cannot freeze the checkpoint.

Acceptance commands:

```bash
uv sync --frozen --extra data
uv run pytest -q tests/unit/test_seen_selection.py tests/integration/test_eval_resume.py
uv run python scripts/eval_seen.py --help
uv run python scripts/select_seen_checkpoint.py --help
```

## Next milestone

On a Linux CUDA VM: 200-step smoke, 100k `seen_expert`, then `eval_seen --run-dir`
and `select_seen_checkpoint.py --write`. Do not tune on target success. Optional
seen LoRA is TODO 25 and must not delay the baseline.
