mod support;

use std::path::PathBuf;

use futures::executor::block_on;
use sipp::backend::set_llama_log_quiet;
use sipp::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode, ModelPlacementConfig,
    NativeRuntimeConfig, ObservabilityRuntimeConfig, PoolingType, ResidencyRuntimeConfig,
    SamplingRuntimeConfig, SchedulerRuntimeConfig,
};
use sipp::{EndpointDescriptor, LocalEmbedOptions, SippClient, SippEmbedRequest};

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let args = support::local_args("SippClient embedding example input.", "embed")?;
        set_llama_log_quiet(true);

        let mut client = SippClient::new();
        client
            .add(
                "default",
                EndpointDescriptor::local(args.model_path, runtime_config(true, None)),
            )
            .await?;

        // Embeddings use the same client as text generation. The local runtime
        // is loaded with `embeddings=true`, and this request asks for a
        // normalized vector.
        let response = client
            .embed(SippEmbedRequest {
                input: args.input,
                local: LocalEmbedOptions {
                    context_key: Some("rust-embed-example".to_string()),
                    normalize: Some(true),
                },
                ..Default::default()
            })
            .await?;

        support::print_embedding(response);
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
            pooling: embeddings.then_some(PoolingType::Mean),
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
