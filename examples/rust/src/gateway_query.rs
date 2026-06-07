mod support;

use std::path::PathBuf;

use cogentlm::backend::set_llama_log_quiet;
use cogentlm::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode, ModelPlacementConfig,
    NativeRuntimeConfig, ObservabilityRuntimeConfig, ResidencyRuntimeConfig, SamplingRuntimeConfig,
    SchedulerRuntimeConfig,
};
use cogentlm::{
    CogentClient, CogentQueryRequest, CogentTextOptions, EndpointDescriptor, GatewayAuthentication,
    GatewayEndpointConfig, GatewayRoutes, GatewaySecret, GatewayTimeoutPolicy, LocalTextOptions,
};
use futures::executor::block_on;

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let args = support::gateway_args(
            "Write one sentence about gateway inference.",
            "gateway_query",
        )?;
        set_llama_log_quiet(true);

        let mut client = CogentClient::new();
        let local_endpoint = client
            .add(
                "local",
                EndpointDescriptor::local(args.model_path, runtime_config(false, None)),
            )
            .await?;
        let config = GatewayEndpointConfig {
            target: args.target.clone(),
            base_url: support::required_env("COGENTLM_GATEWAY_URL")?,
            routes: GatewayRoutes::default(),
            authentication: GatewayAuthentication::Bearer(GatewaySecret::new(
                support::required_env("COGENTLM_GATEWAY_TOKEN")?,
            )),
            static_headers: Default::default(),
            timeouts: GatewayTimeoutPolicy::default(),
            protocol_options: Default::default(),
        };
        let gateway_endpoint = client
            .add("gateway", EndpointDescriptor::gateway(config))
            .await?;

        let local = client
            .query(CogentQueryRequest {
                endpoint: Some(local_endpoint),
                prompt: args.input.clone(),
                options: text_options(),
                local: LocalTextOptions {
                    context_key: Some("rust-gateway-query-local".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?;

        let gateway = client
            .query(CogentQueryRequest {
                endpoint: Some(gateway_endpoint),
                prompt: args.input,
                options: text_options(),
                ..Default::default()
            })
            .await?;

        println!("local:");
        support::print_text(local);
        println!("gateway:");
        support::print_text(gateway);
        Ok(())
    })
}

fn runtime_config(embeddings: bool, projector_path: Option<PathBuf>) -> NativeRuntimeConfig {
    NativeRuntimeConfig {
        placement: ModelPlacementConfig {
            gpu_layers: support::env_parse("COGENTLM_GPU_LAYERS")
                .map(GpuLayerConfig::from_layer_count)
                .unwrap_or(GpuLayerConfig::Auto),
            ..Default::default()
        },
        context: ContextRuntimeConfig {
            n_ctx: support::env_parse("COGENTLM_CONTEXT").or(Some(support::DEFAULT_CONTEXT)),
            n_threads: support::env_parse("COGENTLM_THREADS"),
            n_threads_batch: support::env_parse("COGENTLM_THREADS"),
            embeddings: embeddings.then_some(true),
            ..Default::default()
        },
        sampling: SamplingRuntimeConfig {
            temperature: support::env_parse("COGENTLM_TEMPERATURE")
                .or(Some(support::DEFAULT_TEMPERATURE)),
            seed: support::env_parse("COGENTLM_SEED").or(Some(support::DEFAULT_SEED)),
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
        multimodal: cogentlm::engine::MultimodalRuntimeConfig {
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

fn text_options() -> CogentTextOptions {
    CogentTextOptions {
        max_tokens: support::env_parse("COGENTLM_MAX_TOKENS").or(Some(support::DEFAULT_MAX_TOKENS)),
        temperature: support::env_parse("COGENTLM_TEMPERATURE")
            .or(Some(support::DEFAULT_TEMPERATURE)),
        top_p: support::env_parse("COGENTLM_TOP_P").or(Some(support::DEFAULT_TOP_P)),
        stop: Vec::new(),
    }
}
