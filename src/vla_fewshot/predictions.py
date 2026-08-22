"""Fail-closed lock for the preregistered N=5/10/25 predictions."""

from __future__ import annotations

from pathlib import Path

from vla_fewshot.config import PredictionsLockConfig, load_config
from vla_fewshot.storage.checksums import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "configs" / "predictions.lock.yaml"
DEFAULT_PREDICTIONS = ROOT / "predictions.md"


def load_predictions_lock(path: str | Path = DEFAULT_LOCK) -> PredictionsLockConfig:
    loaded = load_config(path)
    if not isinstance(loaded, PredictionsLockConfig):
        raise TypeError(f"{path} is not a predictions lock")
    return loaded


def require_frozen_predictions(*, root: Path | None = None) -> str:
    """Return the locked SHA-256, or refuse before any target GPU work."""

    project = ROOT if root is None else Path(root)
    lock_path = project / "configs" / "predictions.lock.yaml"
    predictions = project / "predictions.md"
    if not lock_path.is_file():
        raise ValueError(
            "predictions.lock.yaml is missing; freeze predictions.md "
            "before any target fine-tuning. no GPU training was started."
        )
    lock = load_predictions_lock(lock_path)
    if lock.status != "frozen":
        raise ValueError(
            "predictions lock is not frozen. no GPU training was started."
        )
    if not predictions.is_file():
        raise ValueError(
            f"{predictions} is missing. no GPU training was started."
        )
    digest = sha256_file(predictions)
    if digest != lock.sha256:
        raise ValueError(
            f"predictions.md hash {digest} != frozen lock {lock.sha256}. "
            "do not edit numerical claims after the lock; record actuals "
            "in report/report.md. no GPU training was started."
        )
    return digest
