from __future__ import annotations

from sipp import (
    CacheRuntimeConfig,
    ChatMessage,
    SippClient,
    SippTextOptions,
    SippTextRun,
    ContextRuntimeConfig,
    GatewayDescriptor,
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
    read_gateway_args,
    required_env,
)


def runtime_config(*, embeddings: bool) -> NativeRuntimeConfig:
    return NativeRuntimeConfig(
        placement=ModelPlacementConfig(gpu_layers=gpu_layers()),
        context=ContextRuntimeConfig(
            n_ctx=int_env("SIPP_CONTEXT", DEFAULT_CONTEXT),
            n_threads=int_env("SIPP_THREADS"),
            n_threads_batch=int_env("SIPP_THREADS"),
            embeddings=embeddings,
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
        residency=ResidencyRuntimeConfig(max_gpu_models_per_device=1),
        observability=ObservabilityRuntimeConfig(runtime_metrics=True),
    )


def text_options() -> SippTextOptions:
    return SippTextOptions(
        max_tokens=int_env("SIPP_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        temperature=float_env("SIPP_TEMPERATURE", DEFAULT_TEMPERATURE),
        top_p=float_env("SIPP_TOP_P", DEFAULT_TOP_P),
    )


def chat_messages(prompt: str) -> list[ChatMessage]:
    return [
        ChatMessage("system", "Answer concisely."),
        ChatMessage("user", prompt),
    ]


def collect_streamed_text(label: str, run: SippTextRun) -> dict[str, object]:
    streamed = ""
    print(f"{label}_stream=", end="", flush=True)
    for batch in run.tokens():
        print(batch["text"], end="", flush=True)
        streamed += batch["text"]
    print()
    result = run.result()
    if streamed != result["text"]:
        raise RuntimeError("streamed token batches did not match final response text")
    return result


def main() -> None:
    model, target, prompt = read_gateway_args(
        "gateway_chat", "Explain gateway-backed inference in one sentence."
    )
    set_llama_log_quiet(True)

    client = SippClient()
    local_endpoint = client.add(
        "local",
        LocalModelDescriptor(model, runtime_config(embeddings=False)),
    )
    gateway_endpoint = client.add(
        "gateway",
        GatewayDescriptor(
            target,
            required_env("SIPP_GATEWAY_URL"),
            authentication_kind="bearer",
            authentication_value=required_env("SIPP_GATEWAY_TOKEN"),
        )
    )

    local_run = client.chat(
        chat_messages(prompt),
        endpoint=local_endpoint,
        options=text_options(),
        local=LocalTextOptions(context_key="python-gateway-chat-local"),
        emit_tokens=True,
    )
    local = collect_streamed_text("local", local_run)

    gateway_run = client.chat(
        chat_messages(prompt),
        endpoint=gateway_endpoint,
        options=text_options(),
        emit_tokens=True,
    )
    gateway = collect_streamed_text("gateway", gateway_run)

    print("local:")
    print_text(local)
    print("gateway:")
    print_text(gateway)


if __name__ == "__main__":
    main()
