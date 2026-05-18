import os
import sys
from pathlib import Path

_DLL_DIRECTORIES = []


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


_add_windows_dll_directories()

from ._native import (
    CacheRuntimeConfig,
    ChatMessage,
    CogentEngine,
    ContextRuntimeConfig,
    ModelPlacementConfig,
    ModelLoadOptions,
    ModelService,
    MultimodalRuntimeConfig,
    NativeRuntimeConfig,
    ObservabilityRuntimeConfig,
    QueryOptions,
    ResidencyRuntimeConfig,
    SamplingRuntimeConfig,
    SchedulerRuntimeConfig,
    backend_observability_json,
    set_llama_log_quiet,
)

__all__ = [
    "CacheRuntimeConfig",
    "ChatMessage",
    "CogentEngine",
    "ContextRuntimeConfig",
    "ModelPlacementConfig",
    "ModelLoadOptions",
    "ModelService",
    "MultimodalRuntimeConfig",
    "NativeRuntimeConfig",
    "ObservabilityRuntimeConfig",
    "QueryOptions",
    "ResidencyRuntimeConfig",
    "SamplingRuntimeConfig",
    "SchedulerRuntimeConfig",
    "backend_observability_json",
    "set_llama_log_quiet",
]
