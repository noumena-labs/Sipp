from __future__ import annotations

from sipp import (
    CacheRuntimeConfig,
    SippClient,
    ContextRuntimeConfig,
    GatewayDescriptor,
    LocalEmbedOptions,
    LocalModelDescriptor,
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
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    float_env,
    gpu_layers,
    int_env,
    print_embedding,
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
            pooling="mean" if embeddings else None,
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


def main() -> None:
    model, target, input_text = read_gateway_args(
        "gateway_embed", "SippClient gateway embedding example input."
    )
    set_llama_log_quiet(True)

    client = SippClient()
    local_endpoint = client.add(
        "local",
        LocalModelDescriptor(model, runtime_config(embeddings=True)),
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

    local = client.embed(
        input_text,
        endpoint=local_endpoint,
        local=LocalEmbedOptions(
            context_key="python-gateway-embed-local",
            normalize=True,
        ),
    ).result()
    gateway = client.embed(
        input_text,
        endpoint=gateway_endpoint,
    ).result()

    print("local:")
    print_embedding(local)
    print("gateway:")
    print_embedding(gateway)


if __name__ == "__main__":
    main()
