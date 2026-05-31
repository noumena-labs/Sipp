use std::env;
use std::path::PathBuf;

use cogentlm_client::CogentClient;
use cogentlm_engine::backend::set_llama_log_quiet;
use cogentlm_engine::engine::{
    CacheKeyPolicy, CacheRuntimeConfig, ContextRuntimeConfig, GpuLayerConfig, KvReuseMode,
    ModelPlacementConfig, MultimodalRuntimeConfig, NativeRuntimeConfig, ObservabilityRuntimeConfig,
    ResidencyRuntimeConfig, SamplingRuntimeConfig, SchedulerRuntimeConfig,
};

pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

pub struct ExampleArgs {
    pub model_path: PathBuf,
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

pub async fn load_client(model_path: PathBuf, embeddings: bool) -> ExampleResult<CogentClient> {
    set_llama_log_quiet(true);
    let mut client = CogentClient::new();
    client
        .load_model("default", model_path, runtime_config(embeddings))
        .await?;
    Ok(client)
}

fn runtime_config(embeddings: bool) -> NativeRuntimeConfig {
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
            cache_key_policy: CacheKeyPolicy::ContextKey,
            ..Default::default()
        },
        multimodal: MultimodalRuntimeConfig::default(),
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
