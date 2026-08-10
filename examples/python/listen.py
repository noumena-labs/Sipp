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
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python examples/python/listen.py "
            "<model.gguf> <projector.gguf> <audio>"
        )
    model_path, projector_path, audio_path = sys.argv[1:]
    set_llama_log_quiet(True)

    client = SippClient()
    model = client.models.add([model_path, projector_path])
    client.add("asr", Endpoint.local(model, runtime=runtime_config(8192)))
    response = client.listen(
        Path(audio_path).read_bytes(),
        language=os.getenv("SIPP_LANGUAGE"),
    ).result()
    print(response["text"].strip())


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
