import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

_DLL_DIRECTORIES = []
_NATIVE_MODULE_NAME = f"{__name__}._native"
_BACKEND_BINARY_DIR = Path(__file__).resolve().parent / "binaries"
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
    requested = os.environ.get("COGENTLM_PYTHON_BACKEND", "auto").lower()
    if requested not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise RuntimeError(
            f"Invalid COGENTLM_PYTHON_BACKEND={requested}. Expected one of: {valid}"
        )

    return _auto_backends_for_host() if requested == "auto" else [requested]


def _is_python_extension(path: Path) -> bool:
    return path.suffix in {".pyd", ".so"}


def _backend_binary_path(backend: str) -> Optional[Path]:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = _BACKEND_BINARY_DIR / f"_native_{backend}{suffix}"
        if path.is_file():
            return path

    for path in sorted(_BACKEND_BINARY_DIR.glob(f"_native_{backend}*")):
        if path.is_file() and _is_python_extension(path):
            return path

    return None


def _has_staged_backend_binaries() -> bool:
    return any(
        path.is_file() and _is_python_extension(path)
        for path in _BACKEND_BINARY_DIR.glob("_native_*")
    )


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

    path = os.environ.get("COGENTLM_PYTHON_NATIVE_LIBRARY_PATH")
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


def _load_native_module() -> object:
    global _ACTIVE_BACKEND

    explicit = _load_explicit_native_library()
    if explicit is not None:
        return explicit

    requested = os.environ.get("COGENTLM_PYTHON_BACKEND", "auto").lower()
    requested_is_explicit = requested != "auto"
    has_staged_binaries = _has_staged_backend_binaries()
    errors: list[tuple[str, BaseException]] = []
    for backend in _requested_backends():
        path = _backend_binary_path(backend)
        if path is None:
            if requested_is_explicit and has_staged_binaries:
                errors.append(
                    (
                        backend,
                        FileNotFoundError(f"staged {backend} backend binary was not found"),
                    )
                )
            continue

        try:
            module = _load_extension_from_path(path)
            _assert_backend_usable(module, backend)
            _ACTIVE_BACKEND = backend
            return module
        except Exception as error:
            errors.append((backend, error))
            sys.modules.pop(_NATIVE_MODULE_NAME, None)

    if not errors:
        return _load_direct_native_module()

    detail = "\n".join(f"{backend}: {error}" for backend, error in errors)
    raise RuntimeError(
        f"CogentLM failed to load a usable Python backend for {sys.platform} "
        f"{os.name}.\n{detail}"
    ) from errors[-1][1]


def get_active_backend() -> str:
    return _ACTIVE_BACKEND


_add_windows_dll_directories()
_native = _load_native_module()

CacheRuntimeConfig = _native.CacheRuntimeConfig
ChatMessage = _native.ChatMessage
CogentEngine = _native.CogentEngine
ContextRuntimeConfig = _native.ContextRuntimeConfig
DEFAULT_CONTEXT_KEY = _native.DEFAULT_CONTEXT_KEY
DEFAULT_MAX_TOKENS = _native.DEFAULT_MAX_TOKENS
DEFAULT_MODEL_BACKEND = _native.DEFAULT_MODEL_BACKEND
DEFAULT_MODEL_STATS = _native.DEFAULT_MODEL_STATS
ModelPlacementConfig = _native.ModelPlacementConfig
ModelLoadOptions = _native.ModelLoadOptions
ModelService = _native.ModelService
MultimodalRuntimeConfig = _native.MultimodalRuntimeConfig
NativeRuntimeConfig = _native.NativeRuntimeConfig
ObservabilityRuntimeConfig = _native.ObservabilityRuntimeConfig
QueryOptions = _native.QueryOptions
ResidencyRuntimeConfig = _native.ResidencyRuntimeConfig
SamplingRuntimeConfig = _native.SamplingRuntimeConfig
SchedulerPolicyConfig = _native.SchedulerPolicyConfig
SchedulerRuntimeConfig = _native.SchedulerRuntimeConfig
backend_observability_json = _native.backend_observability_json
set_llama_log_quiet = _native.set_llama_log_quiet

__all__ = [
    "CacheRuntimeConfig",
    "ChatMessage",
    "CogentEngine",
    "ContextRuntimeConfig",
    "DEFAULT_CONTEXT_KEY",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_BACKEND",
    "DEFAULT_MODEL_STATS",
    "ModelPlacementConfig",
    "ModelLoadOptions",
    "ModelService",
    "MultimodalRuntimeConfig",
    "NativeRuntimeConfig",
    "ObservabilityRuntimeConfig",
    "QueryOptions",
    "ResidencyRuntimeConfig",
    "SamplingRuntimeConfig",
    "SchedulerPolicyConfig",
    "SchedulerRuntimeConfig",
    "backend_observability_json",
    "get_active_backend",
    "set_llama_log_quiet",
]
