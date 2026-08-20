from pathlib import Path

from vla_fewshot.git_guard import MAX_FILE_BYTES, scan_paths


def test_clean_text_file_passes(tmp_path: Path) -> None:
    path = tmp_path / "clean.py"
    path.write_text("print('safe')\n", encoding="utf-8")
    assert scan_paths([path], tmp_path) == []


def test_probable_secret_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "credentials.txt"
    path.write_text("AWS_ACCESS_KEY_ID=AKIA" + "A" * 16, encoding="utf-8")
    violations = scan_paths([path], tmp_path)
    assert any("AWS access key" in violation.reason for violation in violations)


def test_empty_secret_template_values_are_allowed(tmp_path: Path) -> None:
    path = tmp_path / ".env.example"
    path.write_text(
        "HF_TOKEN=\nAWS_ACCESS_KEY_ID=\nAWS_SECRET_ACCESS_KEY=\n",
        encoding="utf-8",
    )
    assert scan_paths([path], tmp_path) == []


def test_runtime_payload_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    path = run_dir / "metrics.csv"
    path.write_text("loss\n1.0\n", encoding="utf-8")
    violations = scan_paths([path], tmp_path)
    assert any("runtime artifact directory" in violation.reason for violation in violations)


def test_large_file_is_rejected_without_reading_it(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    with path.open("wb") as handle:
        handle.truncate(MAX_FILE_BYTES + 1)
    violations = scan_paths([path], tmp_path)
    assert any("maximum" in violation.reason for violation in violations)
