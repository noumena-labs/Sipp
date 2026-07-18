mod support;

use futures::executor::block_on;
use sipp::backend::set_llama_log_quiet;
use sipp::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode, ModelPlacementConfig,
    NativeRuntimeConfig, ObservabilityRuntimeConfig, PoolingType, ResidencyRuntimeConfig,
    SamplingRuntimeConfig, SchedulerRuntimeConfig,
};
use sipp::{
    GatewayAuthentication, GatewayDescriptor, GatewayRoutes, GatewaySecret, GatewayTimeoutPolicy,
    LocalDescriptor, LocalEmbedOptions, SippClient, SippEmbedRequest,
};

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let args = support::gateway_args(
            "SippClient gateway embedding example input.",
            "gateway_embed",
        )?;
        set_llama_log_quiet(true);

        let mut client = SippClient::new();
        let mut local_descriptor = LocalDescriptor::files([args.model_path]);
        local_descriptor.config = runtime_config(true);
        let local_endpoint = client.add("local", local_descriptor).await?;
        let descriptor = GatewayDescriptor {
            target: args.target.clone(),
            base_url: support::required_env("SIPP_GATEWAY_URL")?,
            routes: GatewayRoutes::default(),
            authentication: GatewayAuthentication::Bearer(GatewaySecret::new(
                support::required_env("SIPP_GATEWAY_TOKEN")?,
            )),
            static_headers: Default::default(),
            timeouts: GatewayTimeoutPolicy::default(),
            protocol_options: Default::default(),
        };
        let gateway_endpoint = client.add("gateway", descriptor).await?;

        let local = client
            .embed(SippEmbedRequest {
                endpoint: Some(local_endpoint),
                input: args.input.clone(),
                local: LocalEmbedOptions {
                    context_key: Some("rust-gateway-embed-local".to_string()),
                    normalize: Some(true),
                },
                ..Default::default()
            })
            .await?;

        let gateway = client
            .embed(SippEmbedRequest {
                endpoint: Some(gateway_endpoint),
                input: args.input,
                ..Default::default()
            })
            .await?;

        println!("local:");
        support::print_embedding(local);
        println!("gateway:");
        support::print_embedding(gateway);
        Ok(())
    })
}

fn runtime_config(embeddings: bool) -> NativeRuntimeConfig {
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
        multimodal: Default::default(),
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
