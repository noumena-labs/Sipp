from __future__ import annotations

from cogentlm import (
    CacheRuntimeConfig,
    ChatMessage,
    CogentClient,
    CogentTextOptions,
    ContextRuntimeConfig,
    LocalModelDescriptor,
    LocalTextOptions,
    ModelPlacementConfig,
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
    read_local_args,
)


def runtime_config(*, embeddings: bool) -> NativeRuntimeConfig:
    return NativeRuntimeConfig(
        placement=ModelPlacementConfig(gpu_layers=gpu_layers()),
        context=ContextRuntimeConfig(
            n_ctx=int_env("COGENTLM_CONTEXT", DEFAULT_CONTEXT),
            n_threads=int_env("COGENTLM_THREADS"),
            n_threads_batch=int_env("COGENTLM_THREADS"),
            embeddings=embeddings,
        ),
        sampling=SamplingRuntimeConfig(
            temperature=float_env("COGENTLM_TEMPERATURE", DEFAULT_TEMPERATURE),
            seed=int_env("COGENTLM_SEED", DEFAULT_SEED),
        ),
        scheduler=SchedulerRuntimeConfig(
            continuous_batching=True,
            prefill_chunk_size=0,
        ),
        cache=CacheRuntimeConfig(mode="live_slot_prefix"),
        residency=ResidencyRuntimeConfig(max_gpu_models_per_device=1),
        observability=ObservabilityRuntimeConfig(runtime_metrics=True),
    )


def text_options() -> CogentTextOptions:
    return CogentTextOptions(
        max_tokens=int_env("COGENTLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        temperature=float_env("COGENTLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        top_p=float_env("COGENTLM_TOP_P", DEFAULT_TOP_P),
    )


def main() -> None:
    model, prompt = read_local_args("chat", "Explain the CogentClient API in one sentence.")
    set_llama_log_quiet(True)

    client = CogentClient()
    client.add(
        "default",
        LocalModelDescriptor(model, runtime_config(embeddings=False)),
    )

    # `chat` sends role-tagged messages and can stream partial token batches.
    run = client.chat(
        [
            ChatMessage("system", "Answer concisely."),
            ChatMessage("user", prompt),
        ],
        options=text_options(),
        local=LocalTextOptions(context_key="python-chat-example"),
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
