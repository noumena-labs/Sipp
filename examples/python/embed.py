from __future__ import annotations

from cogentlm import (
    CacheRuntimeConfig,
    CogentClient,
    ContextRuntimeConfig,
    LocalEmbedOptions,
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


def main() -> None:
    model, input_text = read_local_args("embed", "CogentClient embedding example input.")
    set_llama_log_quiet(True)

    client = CogentClient()
    client.add_local("default", model, runtime_config(embeddings=True))

    # Embeddings use the same local endpoint. The runtime is loaded with
    # embeddings enabled, and the request asks for a normalized vector.
    run = client.embed(
        input_text,
        local=LocalEmbedOptions(
            context_key="python-embed-example",
            normalize=True,
        ),
    )
    print_embedding(run.result())


if __name__ == "__main__":
    main()
