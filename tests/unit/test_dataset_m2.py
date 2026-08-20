from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from tests.helpers.libero_fixture import build_pinned_metadata_fixture
from vla_fewshot.config import DataConfig, load_config
from vla_fewshot.data.download import download_dataset
from vla_fewshot.data.expected import TARGET_TASKS
from vla_fewshot.data.inspection import inspect_revision
from vla_fewshot.data.layout import dataset_revision_root
from vla_fewshot.data.leakage import LeakageError, assert_no_leakage, leakage_checks
from vla_fewshot.data.splits import load_target_splits
from vla_fewshot.data.subset import materialize_logical_subset, nested_ids


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "configs" / "splits" / "target_splits.json"
DATA_YAML = ROOT / "configs" / "data.yaml"


@pytest.fixture()
def metadata_root(tmp_path: Path) -> Path:
    return build_pinned_metadata_fixture(tmp_path / "meta", SPLITS)


def _data_config() -> DataConfig:
    config = load_config(DATA_YAML)
    assert isinstance(config, DataConfig)
    return config


def test_inspection_accepts_spec_counts_and_target_prefixes(metadata_root: Path) -> None:
    config = _data_config()
    report = inspect_revision(
        revision_root=metadata_root,
        repo_id=config.dataset.repo_id,
        revision=config.dataset.revision,
        splits=load_target_splits(SPLITS),
    )
    assert report["acceptance_complete"]
    assert report["videos_decoded"] is False
    bowl = report["target_episode_ids"]["bowl_stove"]
    assert bowl[:5] == [13, 15, 16, 22, 36]
    assert bowl[:10] == bowl[:25][:10]
    assert report["suites"]["libero_90"]["episode_parquet_rows"] == 3921
    assert report["suites"]["libero_goal"]["episode_parquet_rows"] == 428


def test_leakage_fails_when_target_text_is_in_seen(tmp_path: Path) -> None:
    root = build_pinned_metadata_fixture(tmp_path / "meta", SPLITS)
    seen_tasks = root / "libero_90" / "meta" / "tasks.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(seen_tasks)
    rows = table.to_pylist()
    rows[0]["task"] = "put the bowl on the stove"
    pq.write_table(pa.Table.from_pylist(rows), seen_tasks)

    report = leakage_checks(
        revision_root=root,
        data_config=_data_config(),
        splits=load_target_splits(SPLITS),
        stage="seen",
    )
    assert not report["acceptance_complete"]
    with pytest.raises(LeakageError, match="bowl_stove:absent_from_seen"):
        assert_no_leakage(
            revision_root=root,
            data_config=_data_config(),
            splits_path=SPLITS,
            stage="seen",
        )


def test_logical_subset_is_nested_and_refuses_different_overwrite(
    tmp_path: Path,
) -> None:
    splits = load_target_splits(SPLITS)
    first = materialize_logical_subset(
        output_dir=tmp_path / "bowl_n5",
        repo_id="nvidia/LIBERO_LeRobot_v3",
        revision="e5907374380b8f96511957e6ba5582be52a1e179",
        task_slug="bowl_stove",
        n_demos=5,
        splits=splits,
    )
    again = materialize_logical_subset(
        output_dir=tmp_path / "bowl_n5",
        repo_id="nvidia/LIBERO_LeRobot_v3",
        revision="e5907374380b8f96511957e6ba5582be52a1e179",
        task_slug="bowl_stove",
        n_demos=5,
        splits=splits,
    )
    assert first["episode_ids"] == again["episode_ids"] == nested_ids(
        splits.tasks["bowl_stove"].episode_ids_first_25, 5
    )
    with pytest.raises(FileExistsError, match="different subset"):
        materialize_logical_subset(
            output_dir=tmp_path / "bowl_n5",
            repo_id="nvidia/LIBERO_LeRobot_v3",
            revision="e5907374380b8f96511957e6ba5582be52a1e179",
            task_slug="bowl_stove",
            n_demos=10,
            splits=splits,
        )


def test_download_refuses_to_overwrite_a_different_revision(tmp_path: Path) -> None:
    config = _data_config()
    first_root = dataset_revision_root(
        tmp_path, config.dataset.repo_id, config.dataset.revision
    )
    first_root.mkdir(parents=True)
    (first_root / "REVISION").write_text("0" * 40, encoding="utf-8")

    with pytest.raises(FileExistsError, match="already stores revision"):
        download_dataset(
            repo_id=config.dataset.repo_id,
            revision=config.dataset.revision,
            datasets_dir=tmp_path,
            metadata_only=True,
        )


def test_metadata_only_download_rejects_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _data_config()

    def _fake_snapshot(**kwargs) -> None:
        root = Path(kwargs["local_dir"])
        for suite in ("libero_90", "libero_goal"):
            meta = root / suite / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text("{}", encoding="utf-8")
            (meta / "stats.json").write_text("{}", encoding="utf-8")
            (meta / "tasks.parquet").write_bytes(b"parquet")
        video = root / "libero_90" / "videos"
        video.mkdir(parents=True)
        (video / "episode.mp4").write_bytes(b"not-a-video")

    monkeypatch.setattr("vla_fewshot.data.download._snapshot_download", _fake_snapshot)
    with pytest.raises(RuntimeError, match="video files"):
        download_dataset(
            repo_id=config.dataset.repo_id,
            revision=config.dataset.revision,
            datasets_dir=tmp_path,
            metadata_only=True,
        )


def test_video_download_requires_exactly_one_suite(tmp_path: Path) -> None:
    config = _data_config()
    with pytest.raises(ValueError, match="exactly one --suite"):
        download_dataset(
            repo_id=config.dataset.repo_id,
            revision=config.dataset.revision,
            datasets_dir=tmp_path,
            metadata_only=False,
            include_videos=True,
            suites=("libero_90", "libero_goal"),
        )


def test_target_task_contract_matches_data_yaml() -> None:
    config = _data_config()
    for slug, spec in TARGET_TASKS.items():
        assert config.targets[slug].task_text == spec["task_text"]
        assert config.targets[slug].task_index == spec["task_index"]
        assert config.targets[slug].available_count == spec["available_count"]


def test_tasks_parquet_index_level_column_is_accepted(
    tmp_path: Path,
) -> None:
    root = build_pinned_metadata_fixture(tmp_path / "meta", SPLITS)
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = root / "libero_goal" / "meta" / "tasks.parquet"
    rows = pq.read_table(path).to_pylist()
    rewritten = [
        {"task_index": row["task_index"], "__index_level_0__": row["task"]}
        for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(rewritten), path)
    report = inspect_revision(
        revision_root=root,
        repo_id=_data_config().dataset.repo_id,
        revision=_data_config().dataset.revision,
        splits=load_target_splits(SPLITS),
    )
    assert report["acceptance_complete"]
    wine = next(
        item
        for item in report["suites"]["libero_goal"]["task_catalog"]
        if item["task_index"] == 4
    )
    assert wine["task_text"] == "put the wine bottle on top of the cabinet"


def test_train_and_report_scripts_call_the_leakage_gate() -> None:
    for name in ("train_seen.py", "train_target.py", "collect_results.py"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "maybe_assert_no_leakage" in text


def test_leakage_gate_runs_when_metadata_is_present(
    metadata_root: Path, tmp_path: Path
) -> None:
    from vla_fewshot.data.gates import maybe_assert_no_leakage

    config = _data_config()
    encoded = dataset_revision_root(
        tmp_path / "datasets",
        config.dataset.repo_id,
        config.dataset.revision,
    )
    encoded.parent.mkdir(parents=True)
    encoded.symlink_to(metadata_root)
    report = maybe_assert_no_leakage(
        config_path=DATA_YAML,
        splits_path=SPLITS,
        output_root=tmp_path / "datasets",
        stage="seen",
        required=True,
    )
    assert report is not None
    assert report["acceptance_complete"]
