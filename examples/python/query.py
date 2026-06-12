from __future__ import annotations

from sipp import (
    CacheRuntimeConfig,
    SippClient,
    SippTextOptions,
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


def runtime_config(
    *,
    embeddings: bool,
    projector_path: str | None = None,
) -> NativeRuntimeConfig:
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
    model, prompt = read_local_args("query", "Write one sentence about local inference.")
    set_llama_log_quiet(True)

    client = SippClient()
    client.add(
        "default",
        LocalModelDescriptor(model, runtime_config(embeddings=False)),
    )

    # `query` is the simplest text-generation call: one prompt in, one response out.
    run = client.query(
        prompt,
        options=text_options(),
        local=LocalTextOptions(context_key="python-query-example"),
    )
    print_text(run.result())


if __name__ == "__main__":
    main()
