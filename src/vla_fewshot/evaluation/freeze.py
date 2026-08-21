"""Freeze `configs/selected_seen_checkpoint.yaml` after probe selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vla_fewshot.calibration import DEFAULT_SELECTED, load_selected_checkpoint
from vla_fewshot.config import SelectedCheckpointConfig
from vla_fewshot.evaluation.select import SelectionResult
from vla_fewshot.reproducibility import atomic_write_text


class FreezeError(ValueError):
    """Raised when a selected seen checkpoint cannot be frozen."""


def selection_payload(
    current: SelectedCheckpointConfig,
    result: SelectionResult,
    *,
    run_id: str | None,
) -> dict[str, Any]:
    return {
        "kind": "selected_checkpoint",
        "schema_version": 1,
        "status": "frozen",
        "protocol_id": current.protocol_id,
        "tolerance_success": current.tolerance_success,
        "fallback_step": current.fallback_step,
        "selection_rule": current.selection_rule,
        "sha256": result.score.sha256,
        "uri": result.score.uri,
        "run_id": run_id,
        "step": result.score.step,
        "probe_mean_success": result.score.mean_success,
    }


def freeze_selected_checkpoint(
    path: Path,
    result: SelectionResult,
    *,
    run_id: str | None,
    write: bool,
) -> SelectedCheckpointConfig:
    template = path if path.exists() else DEFAULT_SELECTED
    current = load_selected_checkpoint(template)
    payload = selection_payload(current, result, run_id=run_id)
    frozen = SelectedCheckpointConfig.model_validate(payload)
    if current.status == "frozen":
        if current.sha256 != frozen.sha256 or current.step != frozen.step:
            raise FreezeError(
                "selected seen checkpoint is already frozen to a different "
                f"hash/step: {current.sha256}@{current.step} vs "
                f"{frozen.sha256}@{frozen.step}"
            )
        return current
    if write:
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        atomic_write_text(path, text, overwrite=True)
        return load_selected_checkpoint(path)
    return frozen
