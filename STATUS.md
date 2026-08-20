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

## Next milestone

M3 — LIBERO environment adapters, observation parity, state/action/gripper
mapping and expert replay. This stage still does not require buying a GPU
until you want hardware doctor or M5 smoke.
