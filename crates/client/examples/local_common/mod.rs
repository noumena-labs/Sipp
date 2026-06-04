#![allow(dead_code)]

use std::env;
use std::path::PathBuf;

use cogentlm_client::CogentClient;
use cogentlm_engine::backend::set_llama_log_quiet;
use cogentlm_engine::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode, ModelPlacementConfig,
    MultimodalRuntimeConfig, NativeRuntimeConfig, ObservabilityRuntimeConfig,
    ResidencyRuntimeConfig, SamplingRuntimeConfig, SchedulerRuntimeConfig,
};

pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

pub struct ExampleArgs {
    pub model_path: PathBuf,
    pub input: String,
}

pub struct VisionExampleArgs {
    pub model_path: PathBuf,
    pub projector_path: PathBuf,
    pub image_path: PathBuf,
    pub input: String,
}

pub fn args(default_input: &'static str) -> ExampleResult<ExampleArgs> {
    let mut args = env::args().skip(1);
    let model_path = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-client --example <query|chat|embed> -- <model.gguf> [input]",
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

pub fn vision_args(default_input: &'static str) -> ExampleResult<VisionExampleArgs> {
    let mut args = env::args().skip(1);
    let model_path = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-client --example vision_chat -- \
             <model.gguf> <projector.gguf> <image> [input]",
        )
    })?;
    let projector_path = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-client --example vision_chat -- \
             <model.gguf> <projector.gguf> <image> [input]",
        )
    })?;
    let image_path = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-client --example vision_chat -- \
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

pub async fn load_client(model_path: PathBuf, embeddings: bool) -> ExampleResult<CogentClient> {
    load_client_with_projector(model_path, None, embeddings).await
}

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

pub fn env_parse<T>(name: &'static str) -> Option<T>
where
    T: std::str::FromStr,
{
    env::var(name).ok().and_then(|value| value.parse().ok())
}
