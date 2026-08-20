import os
from pathlib import Path

import pytest

from vla_fewshot.config import DataConfig, load_config
from vla_fewshot.data.inspection import inspect_revision
from vla_fewshot.data.layout import dataset_revision_root, resolve_datasets_dir
from vla_fewshot.data.splits import load_target_splits


ROOT = Path(__file__).resolve().parents[2]
RUN_HF = os.environ.get("VLA_RUN_HF_TESTS") == "1"


@pytest.mark.integration
@pytest.mark.huggingface
@pytest.mark.skipif(not RUN_HF, reason="set VLA_RUN_HF_TESTS=1 to download pinned metadata")
def test_live_metadata_matches_spec_split() -> None:
    config = load_config(ROOT / "configs" / "data.yaml")
    assert isinstance(config, DataConfig)
    root = dataset_revision_root(
        resolve_datasets_dir(),
        config.dataset.repo_id,
        config.dataset.revision,
    )
    report = inspect_revision(
        revision_root=root,
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        splits=load_target_splits(ROOT / "configs" / "splits" / "target_splits.json"),
    )
    assert report["acceptance_complete"]
    assert report["videos_decoded"] is False
