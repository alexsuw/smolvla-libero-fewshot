import os
import platform

import pytest

from vla_fewshot.env.libero_env import require_libero_runtime


def test_require_libero_runtime_fails_off_linux() -> None:
    if platform.system() == "Linux":
        pytest.skip("Linux host can import the gpu extra")
    with pytest.raises(RuntimeError, match="Linux EGL"):
        require_libero_runtime()


@pytest.mark.skipif(
    os.environ.get("VLA_RUN_GPU_TESTS") != "1",
    reason="LIBERO expert replay requires Linux gpu extra and EGL",
)
def test_require_libero_runtime_when_requested() -> None:
    require_libero_runtime()
