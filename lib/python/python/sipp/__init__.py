"""Python package facade for Sipp native inference bindings.

The module loads the best available installed native backend, exposes the PyO3
classes used to configure and call Sipp, and reports which backend was selected
for the current process.
"""

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

_DLL_DIRECTORIES = []
_NATIVE_MODULE_NAME = f"{__name__}._native"
_BACKEND_MODULES = {
    "cuda": "sipp_backend_cuda._native",
    "metal": "sipp_backend_metal._native",
    "vulkan": "sipp_backend_vulkan._native",
}
_VALID_BACKENDS = {"auto", "cpu", "cuda", "metal", "vulkan"}
_ACTIVE_BACKEND = "unknown"


def _add_windows_dll_directories() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates = [
        Path(sys.executable).parent,
        Path(sys.base_prefix),
        Path(sys.exec_prefix),
    ]
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value) / "bin")

    seen = set()
    for path in candidates:
        normalized = str(path)
        if normalized in seen or not path.exists():
            continue
        seen.add(normalized)
        _DLL_DIRECTORIES.append(os.add_dll_directory(normalized))


def _auto_backends_for_host() -> list[str]:
    if sys.platform == "darwin":
        return ["metal", "cpu"]
    return ["cuda", "vulkan", "cpu"]


def _requested_backends() -> list[str]:
    requested = os.environ.get("SIPP_PYTHON_BACKEND", "auto").lower()
    if requested not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise RuntimeError(
            f"Invalid SIPP_PYTHON_BACKEND={requested}. Expected one of: {valid}"
        )

    return _auto_backends_for_host() if requested == "auto" else [requested]


def _load_extension_from_path(path: Path) -> object:
    spec = importlib.util.spec_from_file_location(_NATIVE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for {path}")

    previous = sys.modules.pop(_NATIVE_MODULE_NAME, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NATIVE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_NATIVE_MODULE_NAME, None)
        if previous is not None:
            sys.modules[_NATIVE_MODULE_NAME] = previous
        raise
    return module


def _backend_name_matches(value: object, backend: str) -> bool:
    return backend in str(value or "").lower()


def _backend_available(info: dict[str, object], backend: str) -> bool:
    if backend == "cpu":
        return True

    compiled = info.get("compiled")
    available_backends = info.get("availableBackends")
    devices = info.get("devices")
    if not isinstance(compiled, dict):
        compiled = {}
    if not isinstance(available_backends, list):
        available_backends = []
    if not isinstance(devices, list):
        devices = []

    return (
        compiled.get(backend) is True
        and info.get("gpuOffloadSupported") is True
        and (
            any(
                isinstance(item, dict)
                and _backend_name_matches(item.get("name"), backend)
                for item in available_backends
            )
            or any(
                isinstance(item, dict)
                and _backend_name_matches(item.get("backendName"), backend)
                for item in devices
            )
        )
    )


def _assert_backend_usable(module: object, backend: str) -> None:
    if backend == "cpu":
        return

    observability = getattr(module, "backend_observability_json", None)
    if not callable(observability):
        raise RuntimeError(
            f"{backend} binding does not expose backend_observability_json()"
        )

    info = json.loads(observability(True))
    if not isinstance(info, dict) or not _backend_available(info, backend):
        raise RuntimeError(
            f"{backend} binding loaded, but no usable {backend} backend was reported"
        )


def _infer_backend_from_path(path: Path) -> str:
    file_name = path.name.lower()
    for backend in ("cuda", "metal", "vulkan", "cpu"):
        if backend in file_name:
            return backend
    return "cpu"


def _infer_backend_from_module(module: object, path: Optional[Path] = None) -> str:
    fallback = _infer_backend_from_path(path) if path is not None else "cpu"
    observability = getattr(module, "backend_observability_json", None)
    if callable(observability):
        try:
            info = json.loads(observability(False))
            compiled = info.get("compiled") if isinstance(info, dict) else None
            if isinstance(compiled, dict):
                for backend in ("cuda", "metal", "vulkan"):
                    if compiled.get(backend) is True:
                        return backend
        except Exception:
            return fallback

    return fallback


def _load_explicit_native_library() -> Optional[object]:
    global _ACTIVE_BACKEND

    path = os.environ.get("SIPP_PYTHON_NATIVE_LIBRARY_PATH")
    if not path:
        return None

    native_path = Path(path)
    module = _load_extension_from_path(native_path)
    _ACTIVE_BACKEND = _infer_backend_from_module(module, native_path)
    return module


def _load_direct_native_module() -> object:
    global _ACTIVE_BACKEND
    module = importlib.import_module(_NATIVE_MODULE_NAME)
    _ACTIVE_BACKEND = _infer_backend_from_module(module)
    return module


def _is_missing_backend_package(error: ModuleNotFoundError, backend: str) -> bool:
    package_name = f"sipp_backend_{backend}"
    return error.name in {package_name, f"{package_name}._native"}


def _load_backend_package(backend: str) -> Optional[object]:
    global _ACTIVE_BACKEND

    module_name = _BACKEND_MODULES[backend]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if _is_missing_backend_package(error, backend):
            return None
        raise

    _assert_backend_usable(module, backend)
    _ACTIVE_BACKEND = backend
    return module


def _load_explicit_backend(backend: str) -> object:
    global _ACTIVE_BACKEND

    if backend == "cpu":
        return _load_direct_native_module()

    module = _load_backend_package(backend)
    if module is not None:
        return module

    try:
        module = importlib.import_module(_NATIVE_MODULE_NAME)
        _assert_backend_usable(module, backend)
        _ACTIVE_BACKEND = backend
        return module
    except Exception as error:
        raise RuntimeError(
            f"{backend} backend is not installed or usable. "
            f'Install it with: pip install "sipp[{backend}]"'
        ) from error


def _load_native_module() -> object:
    explicit = _load_explicit_native_library()
    if explicit is not None:
        return explicit

    requested = os.environ.get("SIPP_PYTHON_BACKEND", "auto").lower()
    requested_backends = _requested_backends()
    if requested != "auto":
        return _load_explicit_backend(requested_backends[0])

    errors: list[tuple[str, BaseException]] = []
    for backend in requested_backends:
        if backend == "cpu":
            continue

        try:
            module = _load_backend_package(backend)
            if module is not None:
                return module
        except Exception as error:
            errors.append((backend, error))

    try:
        return _load_direct_native_module()
    except Exception as error:
        errors.append(("cpu", error))

    detail = "\n".join(f"{backend}: {error}" for backend, error in errors)
    raise RuntimeError(
        f"Sipp failed to load a usable Python backend for {sys.platform} "
        f"{os.name}.\n{detail}"
    ) from errors[-1][1]


def get_active_backend() -> str:
    """Return the native backend selected by the package loader."""
    return _ACTIVE_BACKEND


_add_windows_dll_directories()
_native = _load_native_module()

CacheRuntimeConfig = _native.CacheRuntimeConfig
ChatMessage = _native.ChatMessage
SippClient = _native.SippClient
SippEmbeddingRun = _native.SippEmbeddingRun
SippTextOptions = _native.SippTextOptions
SippTextRun = _native.SippTextRun
SippTokenIterator = _native.SippTokenIterator
ContextRuntimeConfig = _native.ContextRuntimeConfig
DEFAULT_CONTEXT_KEY = _native.DEFAULT_CONTEXT_KEY
DEFAULT_MAX_TOKENS = _native.DEFAULT_MAX_TOKENS
EndpointRef = _native.EndpointRef
GatewayDescriptor = _native.GatewayDescriptor
LocalEmbedOptions = _native.LocalEmbedOptions
LocalModelDescriptor = _native.LocalModelDescriptor
LocalTextOptions = _native.LocalTextOptions
ModelPlacementConfig = _native.ModelPlacementConfig
MultimodalRuntimeConfig = _native.MultimodalRuntimeConfig
NativeRuntimeConfig = _native.NativeRuntimeConfig
ObservabilityRuntimeConfig = _native.ObservabilityRuntimeConfig
ProviderDescriptor = _native.ProviderDescriptor
ProviderError = _native.ProviderError
EndpointError = _native.EndpointError
ResidencyRuntimeConfig = _native.ResidencyRuntimeConfig
SamplingRuntimeConfig = _native.SamplingRuntimeConfig
SamplingRuntimeOverride = _native.SamplingRuntimeConfig
SchedulerPolicyConfig = _native.SchedulerPolicyConfig
SchedulerRuntimeConfig = _native.SchedulerRuntimeConfig
UnsupportedOperationError = _native.UnsupportedOperationError
backend_observability_json = _native.backend_observability_json
set_llama_log_quiet = _native.set_llama_log_quiet

__all__ = [
    "CacheRuntimeConfig",
    "ChatMessage",
    "SippClient",
    "SippEmbeddingRun",
    "SippTextOptions",
    "SippTextRun",
    "SippTokenIterator",
    "ContextRuntimeConfig",
    "DEFAULT_CONTEXT_KEY",
    "DEFAULT_MAX_TOKENS",
    "EndpointRef",
    "GatewayDescriptor",
    "LocalEmbedOptions",
    "LocalModelDescriptor",
    "LocalTextOptions",
    "ModelPlacementConfig",
    "MultimodalRuntimeConfig",
    "NativeRuntimeConfig",
    "ObservabilityRuntimeConfig",
    "ProviderDescriptor",
    "ProviderError",
    "EndpointError",
    "ResidencyRuntimeConfig",
    "SamplingRuntimeConfig",
    "SamplingRuntimeOverride",
    "SchedulerPolicyConfig",
    "SchedulerRuntimeConfig",
    "UnsupportedOperationError",
    "backend_observability_json",
    "get_active_backend",
    "set_llama_log_quiet",
]
