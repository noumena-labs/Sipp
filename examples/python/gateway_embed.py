from __future__ import annotations

from cogentlm import (
    CacheRuntimeConfig,
    CogentClient,
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


def main() -> None:
    model, alias, input_text = read_gateway_args(
        "gateway_embed", "CogentClient gateway embedding example input."
    )
    set_llama_log_quiet(True)

    client = CogentClient()
    local_endpoint = client.add(
        "local",
        LocalModelDescriptor(model, runtime_config(embeddings=True)),
    )
    gateway = GatewayDescriptor(
        alias,
        required_env("COGENTLM_GATEWAY_URL"),
        required_env("COGENTLM_GATEWAY_TOKEN"),
    )
    gateway_endpoint = client.add("gateway", gateway)

    local = client.embed(
        input_text,
        endpoint=local_endpoint,
        local=LocalEmbedOptions(
            context_key="python-gateway-embed-local",
            normalize=True,
        ),
    ).result()
    remote = client.embed(
        input_text,
        endpoint=gateway_endpoint,
    ).result()

    print("local:")
    print_embedding(local)
    print("gateway:")
    print_embedding(remote)


if __name__ == "__main__":
    main()
