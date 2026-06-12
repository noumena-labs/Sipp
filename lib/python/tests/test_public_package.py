from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_package_import_exposes_public_runtime_helpers() -> None:
    import sipp

    assert callable(sipp.backend_observability_json)
    assert callable(sipp.set_llama_log_quiet)
    assert sipp.get_active_backend() in {"cpu", "cuda", "metal", "vulkan", "unknown"}
    assert hasattr(sipp.SippClient, "add")
    assert hasattr(sipp, "GatewayDescriptor")
    assert not hasattr(sipp.SippClient, "add_" + "local")
    assert not hasattr(sipp.SippClient, "add_http_endpoint")


def test_invalid_backend_environment_is_rejected() -> None:
    env = os.environ.copy()
    env["SIPP_PYTHON_BACKEND"] = "bogus"
    env.pop("SIPP_PYTHON_NATIVE_LIBRARY_PATH", None)

    result = subprocess.run(
        [sys.executable, "-c", "import sipp"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Invalid SIPP_PYTHON_BACKEND=bogus" in f"{result.stdout}\n{result.stderr}"


def test_package_loader_supports_explicit_fake_native_module(tmp_path: Path) -> None:
    fake_native = tmp_path / "fake_native.py"
    fake_native.write_text(
        """
class CacheRuntimeConfig: pass
class ChatMessage: pass
class SippClient: pass
class SippEmbeddingRun: pass
class SippTextOptions: pass
class SippTextRun: pass
class SippTokenIterator: pass
class ContextRuntimeConfig: pass
class EndpointRef: pass
class GatewayDescriptor: pass
class LocalEmbedOptions: pass
class LocalModelDescriptor: pass
class LocalTextOptions: pass
class ModelPlacementConfig: pass
class MultimodalRuntimeConfig: pass
class NativeRuntimeConfig: pass
class ObservabilityRuntimeConfig: pass
class ProviderDescriptor: pass
class ProviderError(Exception): pass
class EndpointError(Exception): pass
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
    env["SIPP_PYTHON_NATIVE_LIBRARY_PATH"] = str(fake_native)
    env.pop("SIPP_PYTHON_BACKEND", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sipp; "
                "assert sipp.get_active_backend() == 'vulkan'; "
                "assert sipp.DEFAULT_CONTEXT_KEY == 'default'; "
                "assert callable(sipp.backend_observability_json); "
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
