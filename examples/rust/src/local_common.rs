//! Shared local-model helpers for Rust example binaries.

use std::env;
use std::path::PathBuf;

use cogentlm::backend::set_llama_log_quiet;
use cogentlm::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode, ModelPlacementConfig,
    MultimodalRuntimeConfig, NativeRuntimeConfig, ObservabilityRuntimeConfig,
    ResidencyRuntimeConfig, SamplingRuntimeConfig, SchedulerRuntimeConfig,
};
use cogentlm::{CogentClient, CogentEmbeddingResponse, CogentTextOptions, CogentTextResponse};

/// Result type used by local Rust examples.
pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Command-line arguments shared by local text and embedding examples.
pub struct ExampleArgs {
    pub model_path: PathBuf,
    pub input: String,
}

/// Command-line arguments for the local multimodal chat example.
pub struct VisionExampleArgs {
    pub model_path: PathBuf,
    pub projector_path: PathBuf,
    pub image_path: PathBuf,
    pub input: String,
}

/// Parse a model path and optional input text for local examples.
pub fn args(default_input: &'static str) -> ExampleResult<ExampleArgs> {
    let mut args = env::args().skip(1);
    let model_path = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-rust-examples --bin <query|chat|embed> -- <model.gguf> [input]",
        )
    })?;
    let input = args.collect::<Vec<_>>().join(" ");
    Ok(ExampleArgs {
        model_path: PathBuf::from(model_path),
        input: if input.is_empty() {
            default_input.to_string()
        } else {
            input
        },
    })
}

/// Parse model, projector, image, and optional input text for vision examples.
pub fn vision_args(default_input: &'static str) -> ExampleResult<VisionExampleArgs> {
    let mut args = env::args().skip(1);
    let model_path = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-rust-examples --bin vision_chat -- \
             <model.gguf> <projector.gguf> <image> [input]",
        )
    })?;
    let projector_path = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-rust-examples --bin vision_chat -- \
             <model.gguf> <projector.gguf> <image> [input]",
        )
    })?;
    let image_path = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-rust-examples --bin vision_chat -- \
             <model.gguf> <projector.gguf> <image> [input]",
        )
    })?;
    let input = args.collect::<Vec<_>>().join(" ");
    Ok(VisionExampleArgs {
        model_path: PathBuf::from(model_path),
        projector_path: PathBuf::from(projector_path),
        image_path: PathBuf::from(image_path),
        input: if input.is_empty() {
            default_input.to_string()
        } else {
            input
        },
    })
}

/// Load a local model into a `CogentClient`.
pub async fn load_client(model_path: PathBuf, embeddings: bool) -> ExampleResult<CogentClient> {
    load_client_with_projector(model_path, None, embeddings).await
}

/// Load a local model and optional multimodal projector into a `CogentClient`.
pub async fn load_client_with_projector(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    embeddings: bool,
) -> ExampleResult<CogentClient> {
    set_llama_log_quiet(true);
    let mut client = CogentClient::new();
    client
        .add_local(
            "default",
            model_path,
            runtime_config(embeddings, projector_path),
        )
        .await?;
    Ok(client)
}

fn runtime_config(embeddings: bool, projector_path: Option<PathBuf>) -> NativeRuntimeConfig {
    NativeRuntimeConfig {
        placement: ModelPlacementConfig {
            gpu_layers: env_parse("COGENTLM_GPU_LAYERS")
                .map(GpuLayerConfig::from_layer_count)
                .unwrap_or(GpuLayerConfig::Auto),
            ..Default::default()
        },
        context: ContextRuntimeConfig {
            n_ctx: env_parse("COGENTLM_CONTEXT"),
            n_threads: env_parse("COGENTLM_THREADS"),
            n_threads_batch: env_parse("COGENTLM_THREADS"),
            embeddings: embeddings.then_some(true),
            ..Default::default()
        },
        sampling: SamplingRuntimeConfig {
            temperature: env_parse("COGENTLM_TEMPERATURE"),
            seed: env_parse("COGENTLM_SEED"),
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
        multimodal: MultimodalRuntimeConfig {
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

/// Build text generation options from the shared example environment variables.
pub fn text_options() -> CogentTextOptions {
    CogentTextOptions {
        max_tokens: env_parse("COGENTLM_MAX_TOKENS"),
        temperature: env_parse("COGENTLM_TEMPERATURE"),
        top_p: env_parse("COGENTLM_TOP_P"),
        stop: Vec::new(),
    }
}

/// Print a text response and local runtime metrics when present.
pub fn print_text(response: CogentTextResponse) {
    println!("endpoint={:?}", response.endpoint);
    println!("finish_reason={}", response.finish_reason.as_str());
    println!("text={}", response.text.trim());
    if let Some(stats) = response.local_stats {
        println!(
            "metrics=ttft_ms:{:?} decode_ms:{:.3} output_tokens:{} e2e_tps:{:?} decode_tps:{:?}",
            stats.ttft_ms,
            stats.decode_ms,
            stats.output_tokens,
            stats.e2e_tokens_per_second,
            stats.decode_tokens_per_second
        );
    }
}

/// Print a compact embedding response preview.
pub fn print_embedding(response: CogentEmbeddingResponse) {
    let preview = response
        .values
        .iter()
        .take(8)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("endpoint={:?}", response.endpoint);
    println!("dimensions={}", response.values.len());
    println!("pooling={:?}", response.pooling);
    println!("normalized={:?}", response.normalized);
    println!("preview=[{preview}]");
}

/// Parse an environment variable into a typed value.
pub fn env_parse<T>(name: &'static str) -> Option<T>
where
    T: std::str::FromStr,
{
    env::var(name).ok().and_then(|value| value.parse().ok())
}
