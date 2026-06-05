from __future__ import annotations

from cogentlm import (
    CacheRuntimeConfig,
    CogentClient,
    CogentTextOptions,
    ContextRuntimeConfig,
    LocalTextOptions,
    ModelPlacementConfig,
    NativeRuntimeConfig,
    ObservabilityRuntimeConfig,
    RemoteGatewayConfig,
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


def main() -> None:
    model, alias, prompt = read_gateway_args(
        "gateway_query", "Write one sentence about gateway inference."
    )
    set_llama_log_quiet(True)

    client = CogentClient()
    local_endpoint = client.add_local("local", model, runtime_config(embeddings=False))
    gateway = RemoteGatewayConfig(
        alias,
        required_env("COGENTLM_GATEWAY_URL"),
        required_env("COGENTLM_GATEWAY_TOKEN"),
    )
    gateway_endpoint = client.add_remote("gateway", gateway)

    # The app only needs the gateway URL, gateway bearer token, and public alias.
    # Provider credentials or local model paths stay in the gateway process.
    local = client.query(
        prompt,
        endpoint=local_endpoint,
        options=text_options(),
        local=LocalTextOptions(context_key="python-gateway-query-local"),
    ).result()
    remote = client.query(
        prompt,
        endpoint=gateway_endpoint,
        options=text_options(),
    ).result()

    print("local:")
    print_text(local)
    print("gateway:")
    print_text(remote)


if __name__ == "__main__":
    main()
