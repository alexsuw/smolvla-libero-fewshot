"""Shared CLI helpers for dataset root resolution."""

from __future__ import annotations

from pathlib import Path

from vla_fewshot.config import DataConfig, load_config
from vla_fewshot.data.layout import dataset_revision_root, resolve_datasets_dir


def load_data_config(path: Path) -> DataConfig:
    config = load_config(path)
    if not isinstance(config, DataConfig):
        raise SystemExit(f"{path} is not a data config")
    return config


def revision_root_from_args(
    *,
    data_config: DataConfig,
    output_root: Path | None,
) -> Path:
    return dataset_revision_root(
        resolve_datasets_dir(output_root),
        data_config.dataset.repo_id,
        data_config.dataset.revision,
    )
