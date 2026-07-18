from __future__ import annotations

import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _fake_native_module_source(observability: str) -> str:
    return f'''
class CacheRuntimeConfig: pass
class ChatMessage: pass
class ManagedModel: pass
class ModelStore: pass
class SippClient:
    add = object()
    models = object()
class SippEmbeddingRun: pass
class SippTextOptions: pass
class SippTextRun: pass
class SippTokenIterator: pass
class ContextRuntimeConfig: pass
class EndpointRef: pass
class EndpointDescriptor:
    local = object()
    gateway = object()
    provider = object()
class LocalEmbedOptions: pass
class LocalTextOptions: pass
class ModelPlacementConfig: pass
class MultimodalRuntimeConfig: pass
class NativeRuntimeConfig: pass
class ObservabilityRuntimeConfig: pass
class ProviderError(Exception): pass
class EndpointError(Exception): pass
class ModelLifecycleError(Exception): pass
class ResidencyRuntimeConfig: pass
class SamplingRuntimeConfig: pass
class SchedulerPolicyConfig: pass
class SchedulerRuntimeConfig: pass
class UnsupportedOperationError(Exception): pass
DEFAULT_CONTEXT_KEY = "default"
DEFAULT_MAX_TOKENS = 128
def backend_observability_json(include_details):
    return {observability!r}
def set_llama_log_quiet(quiet):
    return None
'''


def test_package_import_exposes_public_runtime_helpers() -> None:
    import sipp

    assert callable(sipp.backend_observability_json)
    assert callable(sipp.set_llama_log_quiet)
    assert sipp.get_active_backend() in {"cpu", "cuda", "metal", "vulkan", "unknown"}
    assert hasattr(sipp.SippClient, "add")
    assert hasattr(sipp.SippClient, "remove")
    assert hasattr(sipp.SippClient, "models")
    assert hasattr(sipp, "EndpointDescriptor")
    assert hasattr(sipp.EndpointDescriptor, "local")
    assert hasattr(sipp.EndpointDescriptor, "gateway")
    assert hasattr(sipp.EndpointDescriptor, "provider")
    assert not hasattr(sipp.EndpointDescriptor, "installed")
    assert not hasattr(sipp, "LocalEndpointDescriptor")
    assert not hasattr(sipp, "GatewayEndpointDescriptor")
    assert not hasattr(sipp, "ProviderEndpointDescriptor")
    assert not hasattr(sipp, "ModelSource")
    assert hasattr(sipp.ModelStore, "install_files")
    assert hasattr(sipp.ModelStore, "install_urls")
    assert hasattr(sipp.ModelStore, "list")
    assert hasattr(sipp.ModelStore, "remove")
    assert issubclass(sipp.ModelLifecycleError, Exception)
    assert sipp.SamplingRuntimeOverride is sipp.SamplingRuntimeConfig

    gateway = sipp.EndpointDescriptor.gateway(
        "model-a", "http://127.0.0.1:8080"
    )
    provider = sipp.EndpointDescriptor.provider(
        "openai", "model-a", api_key="test-key"
    )
    assert type(gateway) is sipp.EndpointDescriptor
    assert type(provider) is sipp.EndpointDescriptor
    with pytest.raises(TypeError):
        sipp.EndpointDescriptor()


def test_remote_503_preserves_lifecycle_metadata_after_shared_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sipp

    class ServiceUnavailableHandler(BaseHTTPRequestHandler):
        attempts = 0

        def do_HEAD(self) -> None:
            type(self).attempts += 1
            self.send_response(503)
            self.send_header("Retry-After", "0")
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ServiceUnavailableHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        monkeypatch.chdir(tmp_path)
        host, port = server.server_address
        client = sipp.SippClient()
        with pytest.raises(sipp.ModelLifecycleError) as caught:
            client.models.install_urls([f"http://{host}:{port}/model.gguf"])

        assert caught.value.code == "REMOTE_METADATA_UNAVAILABLE"
        assert caught.value.status == 503
        assert caught.value.retry_after_ms == 0
        assert ServiceUnavailableHandler.attempts == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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
        _fake_native_module_source('{"compiled":{"vulkan":true}}'),
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


def test_package_loader_supports_installed_backend_package(tmp_path: Path) -> None:
    backend_package = tmp_path / "sipp_backend_vulkan"
    backend_package.mkdir()
    (backend_package / "__init__.py").write_text("", encoding="utf-8")
    (backend_package / "_native.py").write_text(
        _fake_native_module_source(
            '{"compiled":{"vulkan":true},'
            '"gpuOffloadSupported":true,'
            '"availableBackends":[{"name":"vulkan"}]}'
        ),
        encoding="utf-8",
    )
    package_root = Path(__file__).resolve().parents[1] / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(package_root)])
    env["SIPP_PYTHON_BACKEND"] = "vulkan"
    env.pop("SIPP_PYTHON_NATIVE_LIBRARY_PATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sipp; "
                "assert sipp.get_active_backend() == 'vulkan'; "
                "assert sipp.DEFAULT_MAX_TOKENS == 128; "
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
