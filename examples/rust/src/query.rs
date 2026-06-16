mod support;

use std::path::PathBuf;

use futures::executor::block_on;
use sipp::backend::set_llama_log_quiet;
use sipp::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode, ModelPlacementConfig,
    NativeRuntimeConfig, ObservabilityRuntimeConfig, ResidencyRuntimeConfig, SamplingRuntimeConfig,
    SchedulerRuntimeConfig,
};
use sipp::{EndpointDescriptor, LocalTextOptions, SippClient, SippQueryRequest, SippTextOptions};

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let args = support::local_args("Write one sentence about local inference.", "query")?;

        // Keep backend logs quiet so the example output focuses on the result.
        set_llama_log_quiet(true);

        // A client can own one or more endpoints. This example adds one local
        // GGUF model and lets requests use it as the default endpoint.
        let mut client = SippClient::new();
        client
            .add(
                "default",
                EndpointDescriptor::local(args.model_path, runtime_config(false, None)),
            )
            .await?;

        // `query` is the simplest text-generation call: one prompt in, one
        // final text response out.
        let response = client
            .query(SippQueryRequest {
                prompt: args.input,
                options: text_options(),
                local: LocalTextOptions {
                    context_key: Some("rust-query-example".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?;

        support::print_text(response);
        Ok(())
    })
}

fn runtime_config(embeddings: bool, projector_path: Option<PathBuf>) -> NativeRuntimeConfig {
    NativeRuntimeConfig {
        placement: ModelPlacementConfig {
            gpu_layers: support::env_parse("SIPP_GPU_LAYERS")
                .map(GpuLayerConfig::from_layer_count)
                .unwrap_or(GpuLayerConfig::Auto),
            ..Default::default()
        },
        context: ContextRuntimeConfig {
            n_ctx: support::env_parse("SIPP_CONTEXT").or(Some(support::DEFAULT_CONTEXT)),
            n_threads: support::env_parse("SIPP_THREADS"),
            n_threads_batch: support::env_parse("SIPP_THREADS"),
            embeddings: embeddings.then_some(true),
            ..Default::default()
        },
        sampling: SamplingRuntimeConfig {
            temperature: support::env_parse("SIPP_TEMPERATURE")
                .or(Some(support::DEFAULT_TEMPERATURE)),
            seed: support::env_parse("SIPP_SEED").or(Some(support::DEFAULT_SEED)),
            ..Default::default()
        },
        scheduler: SchedulerRuntimeConfig {
            continuous_batching: true,
            prefill_chunk_size: 0,
            ..Default::default()
        },
        cache: CacheRuntimeConfig {
            mode: KvReuseMode::LiveSlotPrefix,
            ..Default::default()
        },
        multimodal: sipp::engine::MultimodalRuntimeConfig {
            projector_path: projector_path.map(|path| path.to_string_lossy().into_owned()),
            ..Default::default()
        },
        residency: ResidencyRuntimeConfig {
            max_gpu_models_per_device: 1,
            ..Default::default()
        },
        observability: ObservabilityRuntimeConfig {
            runtime_metrics: true,
            backend_profiling: false,
        },
    }
}

fn text_options() -> SippTextOptions {
    SippTextOptions {
        max_tokens: support::env_parse("SIPP_MAX_TOKENS").or(Some(support::DEFAULT_MAX_TOKENS)),
        temperature: support::env_parse("SIPP_TEMPERATURE").or(Some(support::DEFAULT_TEMPERATURE)),
        top_p: support::env_parse("SIPP_TOP_P").or(Some(support::DEFAULT_TOP_P)),
        stop: Vec::new(),
    }
}
