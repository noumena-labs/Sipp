from __future__ import annotations

from sipp import (
    CacheRuntimeConfig,
    SippClient,
    SippTextOptions,
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


def main() -> None:
    model, target, prompt = read_gateway_args(
        "gateway_query", "Write one sentence about gateway inference."
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

    # The app only needs the gateway URL, gateway bearer token, and public target.
    # Provider credentials or local model paths stay in the gateway process.
    local = client.query(
        prompt,
        endpoint=local_endpoint,
        options=text_options(),
        local=LocalTextOptions(context_key="python-gateway-query-local"),
    ).result()
    gateway = client.query(
        prompt,
        endpoint=gateway_endpoint,
        options=text_options(),
    ).result()

    print("local:")
    print_text(local)
    print("gateway:")
    print_text(gateway)


if __name__ == "__main__":
    main()
