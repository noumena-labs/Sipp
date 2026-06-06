mod support;

use std::path::PathBuf;

use cogentlm::backend::set_llama_log_quiet;
use cogentlm::engine::{
    CacheRuntimeConfig, ChatMessage, ChatRole, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode,
    ModelPlacementConfig, NativeRuntimeConfig, ObservabilityRuntimeConfig, ResidencyRuntimeConfig,
    SamplingRuntimeConfig, SchedulerRuntimeConfig,
};
use cogentlm::{
    CogentChatRequest, CogentClient, CogentTextOptions, EndpointDescriptor, LocalTextOptions,
};
use futures::executor::block_on;
use futures::StreamExt;

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let args = support::local_args("Explain the CogentClient API in one sentence.", "chat")?;
        set_llama_log_quiet(true);

        let mut client = CogentClient::new();
        client
            .add(
                "default",
                EndpointDescriptor::local(args.model_path, runtime_config(false, None)),
            )
            .await?;

        // `chat` accepts structured messages. Token streaming is enabled here
        // so the user can print partial output while the final response is
        // still being assembled.
        let run = client.chat(CogentChatRequest {
            messages: vec![
                ChatMessage::new(ChatRole::System, "Answer concisely."),
                ChatMessage::new(ChatRole::User, args.input),
            ],
            options: text_options(),
            local: LocalTextOptions {
                context_key: Some("rust-chat-example".to_string()),
                ..Default::default()
            },
            emit_tokens: true,
            ..Default::default()
        });

        let (mut tokens, response) = run.into_parts();
        let mut streamed = String::new();
        while let Some(batch) = tokens.next().await {
            print!("{}", batch.text);
            streamed.push_str(&batch.text);
        }
        println!();

        let response = response.await?;
        if streamed != response.text {
            return Err("streamed token batches did not match final response text".into());
        }
        support::print_text(response);
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
