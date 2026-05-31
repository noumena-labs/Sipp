from __future__ import annotations

import os
import sys

from cogentlm import (
    CacheRuntimeConfig,
    CogentClient,
    CogentTextOptions,
    ContextRuntimeConfig,
    EndpointRef,
    ModelPlacementConfig,
    MultimodalRuntimeConfig,
    NativeRuntimeConfig,
    ObservabilityRuntimeConfig,
    RemoteConfig,
    ResidencyRuntimeConfig,
    SamplingRuntimeConfig,
    SchedulerRuntimeConfig,
    set_llama_log_quiet,
)


def read_args(default_input: str) -> tuple[str, str]:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python examples/<query|chat|embed>.py <model.gguf> [input]")
    return sys.argv[1], " ".join(sys.argv[2:]) or default_input


def load_client(model: str, *, embeddings: bool = False) -> CogentClient:
    set_llama_log_quiet(True)
    client = CogentClient()
    client.add_local("default", model, runtime_config(embeddings=embeddings))
    return client


def read_remote_args(default_input: str) -> tuple[str, str]:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python examples/remote_<query|chat|embed>.py <remote-model> [input]"
        )
    return sys.argv[1], " ".join(sys.argv[2:]) or default_input


def add_openai_remote(client: CogentClient, model: str) -> EndpointRef:
    return client.add_remote(
        "openai",
        RemoteConfig.openai(
            model,
            required_env("OPENAI_API_KEY"),
            base_url=os.getenv("COGENTLM_OPENAI_BASE_URL"),
        ),
    )


def text_options() -> CogentTextOptions:
    return CogentTextOptions(
        max_tokens=int_env("COGENTLM_MAX_TOKENS", 128),
        temperature=float_env("COGENTLM_TEMPERATURE", 0.7),
        top_p=float_env("COGENTLM_TOP_P", 0.8),
    )


def print_text(result: dict[str, object]) -> None:
    print(f"endpoint={result['endpoint']}")
    print(f"finish_reason={result['finish_reason']}")
    print(f"text={str(result['text']).strip()}")
    stats = result["local_stats"]
    if isinstance(stats, dict):
        print(
            "metrics="
            f"ttft_ms:{stats['ttft_ms']} "
            f"decode_ms:{stats['decode_ms']:.3f} "
            f"output_tokens:{stats['output_tokens']} "
            f"tps:{stats['tokens_per_second']}"
        )


def print_embedding(result: dict[str, object]) -> None:
    values = result["values"]
    if not isinstance(values, list):
        raise TypeError("embedding values must be a list")
    preview = ", ".join(f"{float(value):.6f}" for value in values[:8])
    print(f"endpoint={result['endpoint']}")
    print(f"dimensions={len(values)}")
    print(f"pooling={result['pooling']}")
    print(f"normalized={result['normalized']}")
    print(f"preview=[{preview}]")


def runtime_config(*, embeddings: bool) -> NativeRuntimeConfig:
    return NativeRuntimeConfig(
        placement=ModelPlacementConfig(gpu_layers=gpu_layers()),
        context=ContextRuntimeConfig(
            n_ctx=int_env("COGENTLM_CONTEXT", 2048),
            n_threads=int_env("COGENTLM_THREADS"),
            n_threads_batch=int_env("COGENTLM_THREADS"),
            embeddings=embeddings,
        ),
        sampling=SamplingRuntimeConfig(
            temperature=float_env("COGENTLM_TEMPERATURE", 0.7),
            seed=int_env("COGENTLM_SEED", 42),
        ),
        scheduler=SchedulerRuntimeConfig(
            continuous_batching=True,
            prefill_chunk_size=0,
        ),
        cache=CacheRuntimeConfig(
            mode="live_slot_prefix",
            cache_key_policy="context_key",
        ),
        multimodal=MultimodalRuntimeConfig(),
        residency=ResidencyRuntimeConfig(max_gpu_models_per_device=1),
        observability=ObservabilityRuntimeConfig(runtime_metrics=True),
    )


def gpu_layers() -> str | dict[str, int] | None:
    value = os.getenv("COGENTLM_GPU_LAYERS")
    if value in {"all", "auto"}:
        return value
    return None if value is None else {"count": int(value)}


def int_env(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return default if value is None else int(value)


def float_env(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    return default if value is None else float(value)


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
