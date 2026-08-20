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

## Next milestone

M1 — verify exact Python/CUDA/LeRobot/MuJoCo pins on Linux, implement bootstrap
and `doctor.py`, then validate RTX 6000 Blackwell GPU visibility, BF16, EGL,
ffmpeg and durable/object storage.
