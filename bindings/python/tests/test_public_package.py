from __future__ import annotations

import os
import subprocess
import sys


def test_package_import_exposes_public_runtime_helpers() -> None:
    import cogentlm

    assert callable(cogentlm.backend_observability_json)
    assert callable(cogentlm.set_llama_log_quiet)
    assert cogentlm.get_active_backend() in {"cpu", "cuda", "metal", "vulkan", "unknown"}


def test_invalid_backend_environment_is_rejected() -> None:
    env = os.environ.copy()
    env["COGENTLM_PYTHON_BACKEND"] = "bogus"
    env.pop("COGENTLM_PYTHON_NATIVE_LIBRARY_PATH", None)

    result = subprocess.run(
        [sys.executable, "-c", "import cogentlm"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Invalid COGENTLM_PYTHON_BACKEND=bogus" in f"{result.stdout}\n{result.stderr}"
