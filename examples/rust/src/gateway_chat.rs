mod support;

use std::path::PathBuf;

use cogentlm::backend::set_llama_log_quiet;
use cogentlm::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode, ModelPlacementConfig,
    NativeRuntimeConfig, ObservabilityRuntimeConfig, ResidencyRuntimeConfig, SamplingRuntimeConfig,
    SchedulerRuntimeConfig,
};
use cogentlm::engine::{ChatMessage, ChatRole};
use cogentlm::{
    CogentChatRequest, CogentClient, CogentTextOptions, CogentTextResponse, CogentTextRun,
    EndpointDescriptor, LocalTextOptions,
};
use cogentlm::{RemoteGatewayConfig, RemoteSecret};
use futures::executor::block_on;
use futures::StreamExt;

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let args = support::gateway_args(
            "Explain gateway-backed inference in one sentence.",
            "gateway_chat",
        )?;
        set_llama_log_quiet(true);

        let mut client = CogentClient::new();
        let local_endpoint = client
            .add(
                "local",
                EndpointDescriptor::local(args.model_path, runtime_config(false, None)),
            )
            .await?;
        let config = RemoteGatewayConfig {
            alias: args.alias.clone(),
            base_url: support::required_env("COGENTLM_GATEWAY_URL")?,
            token: RemoteSecret::new(support::required_env("COGENTLM_GATEWAY_TOKEN")?),
            timeout: None,
        };
        let gateway_endpoint = client
            .add("gateway", EndpointDescriptor::gateway(config))
            .await?;

        let local_run = client.chat(CogentChatRequest {
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

        let gateway_run = client.chat(CogentChatRequest {
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
    run: CogentTextRun,
) -> support::ExampleResult<CogentTextResponse> {
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
