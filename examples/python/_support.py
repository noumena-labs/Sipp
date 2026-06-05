from __future__ import annotations

import os
import sys

DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_CONTEXT = 2048
DEFAULT_SEED = 42


def read_local_args(command: str, default_input: str) -> tuple[str, str]:
    if len(sys.argv) < 2:
        raise SystemExit(
            f"usage: python examples/python/{command}.py <model.gguf> [input]"
        )
    return sys.argv[1], " ".join(sys.argv[2:]) or default_input


def read_vision_args(default_input: str) -> tuple[str, str, str, str]:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: python examples/python/vision_chat.py "
            "<model.gguf> <projector.gguf> <image> [input]"
        )
    return (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        " ".join(sys.argv[4:]) or default_input,
    )


def read_gateway_args(command: str, default_input: str) -> tuple[str, str, str]:
    if len(sys.argv) < 3:
        raise SystemExit(
            f"usage: python examples/python/{command}.py "
            "<model.gguf> <gateway-alias> [input]"
        )
    return sys.argv[1], sys.argv[2], " ".join(sys.argv[3:]) or default_input


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def int_env(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return default if value is None else int(value)


def float_env(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    return default if value is None else float(value)


def gpu_layers() -> str | dict[str, int] | None:
    value = os.getenv("COGENTLM_GPU_LAYERS")
    if value in {"all", "auto"}:
        return value
    return None if value is None else {"count": int(value)}


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
            f"e2e_tps:{stats['e2e_tokens_per_second']} "
            f"decode_tps:{stats['decode_tokens_per_second']}"
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
