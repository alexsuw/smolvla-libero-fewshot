from pathlib import Path

import pytest

from vla_fewshot.storage.roundtrip import filesystem_roundtrip


def test_filesystem_roundtrip_verifies_and_cleans_probe(tmp_path: Path) -> None:
    result = filesystem_roundtrip(tmp_path, payload_size=128)
    assert result.verified
    assert result.bytes_written == 128
    assert len(result.sha256) == 64
    assert list((tmp_path / ".vla-doctor").iterdir()) == []


def test_filesystem_roundtrip_rejects_empty_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        filesystem_roundtrip(tmp_path, payload_size=0)
