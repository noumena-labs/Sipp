mod support;

use std::path::PathBuf;

use futures::executor::block_on;
use futures::StreamExt;
use sipp::backend::set_llama_log_quiet;
use sipp::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode, ModelPlacementConfig,
    NativeRuntimeConfig, ObservabilityRuntimeConfig, ResidencyRuntimeConfig, SamplingRuntimeConfig,
    SchedulerRuntimeConfig,
};
use sipp::engine::{ChatMessage, ChatRole};
use sipp::{
    EndpointDescriptor, GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes, GatewaySecret,
    GatewayTimeoutPolicy, LocalTextOptions, SippChatRequest, SippClient, SippTextOptions,
    SippTextResponse, SippTextRun,
};

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let args = support::gateway_args(
            "Explain gateway-backed inference in one sentence.",
            "gateway_chat",
        )?;
        set_llama_log_quiet(true);

        let mut client = SippClient::new();
        let local_endpoint = client
            .add(
                "local",
                EndpointDescriptor::local(args.model_path, runtime_config(false, None)),
            )
            .await?;
        let config = GatewayEndpointConfig {
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
        let gateway_endpoint = client
            .add("gateway", EndpointDescriptor::gateway(config))
            .await?;

        let local_run = client.chat(SippChatRequest {
            endpoint: Some(local_endpoint),
            messages: chat_messages(args.input.clone()),
            options: text_options(),
            local: LocalTextOptions {
                context_key: Some("rust-gateway-chat-local".to_string()),
                ..Default::default()
            },
            emit_tokens: true,
            ..Default::default()
        });
        let local = collect_streamed_text("local", local_run).await?;

        let gateway_run = client.chat(SippChatRequest {
            endpoint: Some(gateway_endpoint),
            messages: chat_messages(args.input),
            options: text_options(),
            emit_tokens: true,
            ..Default::default()
        });
        let gateway = collect_streamed_text("gateway", gateway_run).await?;

        println!("local:");
        support::print_text(local);
        println!("gateway:");
        support::print_text(gateway);
        Ok(())
    })
}

fn chat_messages(input: String) -> Vec<ChatMessage> {
    vec![
        ChatMessage::new(ChatRole::System, "Answer concisely."),
        ChatMessage::new(ChatRole::User, input),
    ]
}

async fn collect_streamed_text(
    label: &str,
    run: SippTextRun,
) -> support::ExampleResult<SippTextResponse> {
    let (mut tokens, response) = run.into_parts();
    let mut streamed = String::new();
    print!("{label}_stream=");
    while let Some(batch) = tokens.next().await {
        print!("{}", batch.text);
        streamed.push_str(&batch.text);
    }
    println!();

    let response = response.await?;
    if streamed != response.text {
        return Err("streamed token batches did not match final response text".into());
    }
    Ok(response)
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
