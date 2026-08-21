import os
import platform

import pytest

from vla_fewshot.model.smolvla import require_smolvla_runtime


def test_require_smolvla_runtime_fails_off_linux() -> None:
    if platform.system() == "Linux":
        pytest.skip("Linux host can import the gpu extra")
    with pytest.raises(RuntimeError, match="Linux"):
        require_smolvla_runtime()


@pytest.mark.skipif(
    os.environ.get("VLA_RUN_GPU_TESTS") != "1",
    reason="pinned SmolVLA load requires Linux gpu extra and CUDA",
)
def test_load_pinned_smolvla_when_requested() -> None:
    require_smolvla_runtime()
