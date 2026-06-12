from __future__ import annotations

from pathlib import Path

from sipp import (
    CacheRuntimeConfig,
    ChatMessage,
    SippClient,
    SippTextOptions,
    ContextRuntimeConfig,
    LocalModelDescriptor,
    LocalTextOptions,
    ModelPlacementConfig,
    MultimodalRuntimeConfig,
    NativeRuntimeConfig,
    ObservabilityRuntimeConfig,
    ResidencyRuntimeConfig,
    SamplingRuntimeConfig,
    SchedulerRuntimeConfig,
    set_llama_log_quiet,
)

from _support import (
    DEFAULT_CONTEXT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    float_env,
    gpu_layers,
    int_env,
    print_text,
    read_vision_args,
)


def runtime_config(projector_path: str) -> NativeRuntimeConfig:
    return NativeRuntimeConfig(
        placement=ModelPlacementConfig(gpu_layers=gpu_layers()),
        context=ContextRuntimeConfig(
            n_ctx=int_env("SIPP_CONTEXT", DEFAULT_CONTEXT),
            n_threads=int_env("SIPP_THREADS"),
            n_threads_batch=int_env("SIPP_THREADS"),
        ),
        sampling=SamplingRuntimeConfig(
            temperature=float_env("SIPP_TEMPERATURE", DEFAULT_TEMPERATURE),
            seed=int_env("SIPP_SEED", DEFAULT_SEED),
        ),
        scheduler=SchedulerRuntimeConfig(
            continuous_batching=True,
            prefill_chunk_size=0,
        ),
        cache=CacheRuntimeConfig(mode="live_slot_prefix"),
        multimodal=MultimodalRuntimeConfig(projector_path=projector_path),
        residency=ResidencyRuntimeConfig(max_gpu_models_per_device=1),
        observability=ObservabilityRuntimeConfig(runtime_metrics=True),
    )


def text_options() -> SippTextOptions:
    return SippTextOptions(
        max_tokens=int_env("SIPP_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        temperature=float_env("SIPP_TEMPERATURE", DEFAULT_TEMPERATURE),
        top_p=float_env("SIPP_TOP_P", DEFAULT_TOP_P),
    )


def main() -> None:
    model, projector, image, prompt = read_vision_args("Describe this image in one sentence.")
    set_llama_log_quiet(True)

    client = SippClient()
    client.add("default", LocalModelDescriptor(model, runtime_config(projector)))

    # Multimodal chat uses the same chat API. The projector is in runtime
    # config; image bytes are passed on the request.
    run = client.chat(
        [
            ChatMessage("user", prompt),
        ],
        options=text_options(),
        local=LocalTextOptions(
            context_key="python-vision-chat-example",
            media=[Path(image).read_bytes()],
        ),
        emit_tokens=True,
    )
    streamed = ""
    for batch in run.tokens():
        print(batch["text"], end="", flush=True)
        streamed += batch["text"]
    print()
    result = run.result()
    if streamed != result["text"]:
        raise RuntimeError("streamed token batches did not match final response text")
    print_text(result)


if __name__ == "__main__":
    main()
