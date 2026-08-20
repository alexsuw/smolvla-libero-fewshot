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
