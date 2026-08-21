from pathlib import Path

import pytest

from vla_fewshot.storage.object_store import FileObjectStore
from vla_fewshot.storage.object_sync import (
    COMPLETED_NAME,
    execute_object_sync,
    verify_completed_backup,
)
from vla_fewshot.storage.uri import ObjectUriError, parse_object_uri


ROOT = Path(__file__).resolve().parents[2]


def test_parse_s3_and_file_uris(tmp_path: Path) -> None:
    s3 = parse_object_uri("s3://my-bucket/project/prefix")
    assert s3.scheme == "s3"
    assert s3.bucket == "my-bucket"
    assert s3.prefix == "project/prefix"
    file_loc = parse_object_uri(f"file://{tmp_path}")
    assert file_loc.scheme == "file"
    assert Path(file_loc.prefix) == tmp_path


def test_unsupported_object_scheme_is_rejected() -> None:
    with pytest.raises(ObjectUriError, match="unsupported"):
        parse_object_uri("https://example.invalid/bucket")


def test_object_sync_is_dry_run_first_and_never_deletes(tmp_path: Path) -> None:
    source = tmp_path / "src"
    dest = tmp_path / "obj"
    source.mkdir()
    (source / "weights.json").write_text('{"ok": true}\n', encoding="utf-8")
    uri = f"file://{dest}"
    dry = execute_object_sync(source, uri, execute=False)
    assert dry["dry_run"] is True
    assert dry["deleted"] == 0
    assert not dest.exists()

    report = execute_object_sync(
        source,
        uri,
        execute=True,
        backup_status_path=tmp_path / "backup_status.json",
    )
    assert report["dry_run"] is False
    assert report["deleted"] == 0
    assert report["verified"] is True
    store = FileObjectStore(dest)
    assert store.exists("weights.json")
    assert store.exists(COMPLETED_NAME)
    assert (dest / "_tmp").is_dir()
    verified = verify_completed_backup(store, source=source)
    assert verified["verified"] is True
    assert (tmp_path / "backup_status.json").is_file()


def test_object_sync_refuses_conflicting_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "src"
    dest = tmp_path / "obj"
    source.mkdir()
    dest.mkdir()
    (source / "a.txt").write_text("one\n", encoding="utf-8")
    (dest / "a.txt").write_text("two\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        execute_object_sync(source, f"file://{dest}", execute=True)


def test_sync_cli_has_no_delete_flag() -> None:
    text = (ROOT / "scripts" / "sync_artifacts.py").read_text(encoding="utf-8")
    assert "--delete" not in text
    assert "Never deletes" in text


def test_s3_execute_without_boto3_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("x\n", encoding="utf-8")
    dry = execute_object_sync(source, "s3://example-bucket/prefix", execute=False)
    assert dry["dry_run"] is True
    assert dry["copied"] == 1
    try:
        import boto3  # noqa: F401
    except ImportError:
        with pytest.raises(Exception, match="boto3"):
            execute_object_sync(source, "s3://example-bucket/prefix", execute=True)
    else:
        pytest.skip("boto3 is installed; live S3 execute is not part of CPU tests")
