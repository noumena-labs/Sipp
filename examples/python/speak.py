from __future__ import annotations

import os
import sys
from pathlib import Path

from sipp import (
    ContextRuntimeConfig,
    Endpoint,
    ModelPlacementConfig,
    NativeRuntimeConfig,
    SippClient,
    set_llama_log_quiet,
)

from _support import gpu_layers, int_env


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: python examples/python/speak.py "
            "<model.gguf> <projector.gguf> <output.wav> [text]"
        )
    model_path, projector_path, output_path, *words = sys.argv[1:]
    set_llama_log_quiet(True)

    client = SippClient()
    model = client.models.add([model_path, projector_path])
    client.add("tts", Endpoint.local(model, runtime=runtime_config(4096)))
    speaker_path = os.getenv("SIPP_SPEAKER_AUDIO")
    response = client.speak(
        " ".join(words) or "Hello from Sipp.",
        language=os.getenv("SIPP_LANGUAGE"),
        speaker_audio=Path(speaker_path).read_bytes() if speaker_path else None,
        max_duration_ms=int_env("SIPP_MAX_DURATION_MS"),
    ).result()
    Path(output_path).write_bytes(response["audio"])
    print(
        f'wrote {response["duration_ms"]} ms at '
        f'{response["sample_rate_hz"]} Hz to {output_path}'
    )


def runtime_config(context_size: int) -> NativeRuntimeConfig:
    return NativeRuntimeConfig(
        placement=ModelPlacementConfig(gpu_layers=gpu_layers()),
        context=ContextRuntimeConfig(
            n_ctx=int_env("SIPP_CONTEXT", context_size),
            n_threads=int_env("SIPP_THREADS"),
            n_threads_batch=int_env("SIPP_THREADS"),
            warmup=False,
        ),
    )


if __name__ == "__main__":
    main()
