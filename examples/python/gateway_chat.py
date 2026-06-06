from __future__ import annotations

from cogentlm import (
    CacheRuntimeConfig,
    ChatMessage,
    CogentClient,
    CogentTextOptions,
    CogentTextRun,
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


def chat_messages(prompt: str) -> list[ChatMessage]:
    return [
        ChatMessage("system", "Answer concisely."),
        ChatMessage("user", prompt),
    ]


def collect_streamed_text(label: str, run: CogentTextRun) -> dict[str, object]:
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
    model, alias, prompt = read_gateway_args(
        "gateway_chat", "Explain gateway-backed inference in one sentence."
    )
    set_llama_log_quiet(True)

    client = CogentClient()
    local_endpoint = client.add(
        "local",
        LocalModelDescriptor(model, runtime_config(embeddings=False)),
    )
    gateway = GatewayDescriptor(
        alias,
        required_env("COGENTLM_GATEWAY_URL"),
        required_env("COGENTLM_GATEWAY_TOKEN"),
    )
    gateway_endpoint = client.add("gateway", gateway)

    # Local and gateway chat use the same message and streaming shape.
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
    remote = collect_streamed_text("gateway", gateway_run)

    print("local:")
    print_text(local)
    print("gateway:")
    print_text(remote)


if __name__ == "__main__":
    main()
