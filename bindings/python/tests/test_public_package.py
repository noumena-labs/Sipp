from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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


def test_package_loader_supports_explicit_fake_native_module(tmp_path: Path) -> None:
    fake_native = tmp_path / "fake_native.py"
    fake_native.write_text(
        """
class CacheRuntimeConfig: pass
class ChatMessage: pass
class CogentClient: pass
class CogentEmbeddingRun: pass
class CogentTextOptions: pass
class CogentTextRun: pass
class CogentTokenIterator: pass
class ContextRuntimeConfig: pass
class EndpointRef: pass
class LocalEmbedOptions: pass
class LocalTextOptions: pass
class ModelPlacementConfig: pass
class MultimodalRuntimeConfig: pass
class NativeRuntimeConfig: pass
class ObservabilityRuntimeConfig: pass
class RemoteGatewayConfig: pass
class RemoteError(Exception): pass
class ResidencyRuntimeConfig: pass
class SamplingRuntimeConfig: pass
class SchedulerPolicyConfig: pass
class SchedulerRuntimeConfig: pass
class UnsupportedOperationError(Exception): pass
DEFAULT_CONTEXT_KEY = "default"
DEFAULT_MAX_TOKENS = 128
def backend_observability_json(include_details):
    return '{"compiled":{"vulkan":true}}'
def set_llama_log_quiet(quiet):
    return None
""",
        encoding="utf-8",
    )
    package_root = Path(__file__).resolve().parents[1] / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root)
    env["COGENTLM_PYTHON_NATIVE_LIBRARY_PATH"] = str(fake_native)
    env.pop("COGENTLM_PYTHON_BACKEND", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import cogentlm; "
                "assert cogentlm.get_active_backend() == 'vulkan'; "
                "assert cogentlm.DEFAULT_CONTEXT_KEY == 'default'; "
                "assert callable(cogentlm.backend_observability_json); "
                "print('ok')"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "ok" in result.stdout
