# Integration tests

M1+ tests that require pinned external runtimes, datasets, object storage, EGL,
or GPU resources live here and use explicit pytest markers.

M2 live metadata:

```bash
export VLA_DATASETS_DIR="$HOME/.cache/vla-fewshot/datasets"
export VLA_RUN_HF_TESTS=1
uv run python scripts/download_dataset.py --output-root "$VLA_DATASETS_DIR"
uv run pytest -q tests/integration/test_m2_metadata.py
```

This download is metadata-only and CPU-only. It is not a GPU purchase gate.

M3 live expert replay (Linux GPU VM, after `uv sync --frozen --extra gpu`):

```bash
export VLA_DATASETS_DIR="$HOME/.cache/vla-fewshot/datasets"
export VLA_RUN_GPU_TESTS=1
uv run python scripts/download_dataset.py --include-actions --output-root "$VLA_DATASETS_DIR"
uv run python scripts/check_observation_parity.py --with-env \
  --output-dir artifacts/validation/M3/parity-env
uv run python scripts/replay_expert.py --all-gate --save-video \
  --output-dir artifacts/validation/M3/replay
```

M4 pinned SmolVLA load (Linux GPU VM, after `uv sync --frozen --extra gpu`):

```bash
export VLA_RUN_GPU_TESTS=1
uv run python scripts/smoke_inference.py --config configs/train/smoke.yaml \
  --profile full --output-dir artifacts/validation/M4/full
```
