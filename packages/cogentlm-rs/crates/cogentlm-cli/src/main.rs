use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context};
use clap::Parser;
use cogentlm_engine::engine::{GpuLayerConfig, NativeRuntimeConfig, SamplingRuntimeConfig};
use cogentlm_engine::runtime::request::{GenerateResponseStatus, GenerateTokenEmissionMode};
use cogentlm_engine::runtime::{InferenceRuntime, RequestStepResult};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "cogentlm")]
#[command(about = "CogentLM Rust runtime proof-of-concept CLI")]
struct Args {
    /// Path to a GGUF model.
    model: PathBuf,

    /// Prompt text.
    prompt: String,

    /// Maximum generated tokens.
    #[arg(long, default_value_t = 64)]
    max_tokens: u32,

    /// Context size in tokens.
    #[arg(long, default_value_t = 2048)]
    ctx_size: u32,

    /// Decode batch size in tokens.
    #[arg(long, default_value_t = 512)]
    batch_size: u32,

    /// Number of model layers to offload to GPU.
    #[arg(long, default_value_t = 0)]
    gpu_layers: i32,

    /// Number of generation threads. Zero lets llama.cpp choose.
    #[arg(long, default_value_t = 0)]
    threads: i32,

    /// Sampling temperature. Use 0 for greedy decoding.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-k sampling cutoff.
    #[arg(long, default_value_t = 40)]
    top_k: i32,

    /// Top-p sampling cutoff.
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Min-p sampling cutoff.
    #[arg(long, default_value_t = 0.05)]
    min_p: f32,

    /// Sampling RNG seed.
    #[arg(long, default_value_t = u32::MAX)]
    seed: u32,

    /// Render the prompt as a single user chat message before generation.
    #[arg(long)]
    chat: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut stdout = io::stdout().lock();
    run_native_runtime(&args, &mut stdout)
        .with_context(|| format!("native runtime failed for {}", args.model.display()))?;

    writeln!(stdout)?;
    Ok(())
}

fn run_native_runtime(args: &Args, stdout: &mut impl Write) -> anyhow::Result<()> {
    let mut config = NativeRuntimeConfig::default();
    config.context.n_ctx = Some(args.ctx_size.min(i32::MAX as u32) as i32);
    config.context.n_batch = Some(args.batch_size.min(i32::MAX as u32) as i32);
    config.context.n_ubatch = Some(args.batch_size.min(i32::MAX as u32) as i32);
    config.context.n_parallel = Some(1);
    config.context.n_threads = Some(args.threads);
    config.context.n_threads_batch = Some(args.threads);
    config.placement.gpu_layers = cli_gpu_layers(args.gpu_layers);
    config.sampling = SamplingRuntimeConfig {
        temperature: Some(args.temperature),
        top_k: Some(args.top_k),
        top_p: Some(args.top_p),
        min_p: Some(args.min_p),
        seed: (args.seed != u32::MAX).then_some(args.seed),
        ..SamplingRuntimeConfig::default()
    };
    if args.temperature <= 0.0 {
        config.sampling.top_k = Some(1);
    }

    let mut runtime = InferenceRuntime::load(&args.model, config)?;
    let prompt = if args.chat {
        let messages = json!([{ "role": "user", "content": args.prompt }]);
        let rendered = runtime.apply_chat_template_json(&messages.to_string(), true)?;
        if rendered.is_empty() {
            bail!("model did not provide a usable chat template");
        }
        rendered
    } else {
        args.prompt.clone()
    };

    let request_id = runtime.enqueue_request(
        "",
        prompt,
        args.max_tokens.min(i32::MAX as u32) as i32,
        "",
        "",
        Vec::new(),
        None,
        GenerateTokenEmissionMode::None,
    )?;

    for _ in 0..10_000 {
        let burst = runtime.run_scheduler_burst(256, 1, 0, Duration::ZERO);
        if let Some(response) = runtime.try_peek_completed_response(request_id) {
            if response.status == GenerateResponseStatus::Completed {
                stdout.write_all(response.output_text.as_bytes())?;
                return Ok(());
            }
            bail!(
                "request {} finished with {:?}: {}",
                request_id,
                response.status,
                response.error_message
            );
        }
        if matches!(
            burst.status,
            RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
        ) {
            bail!("scheduler stopped with {:?}", burst.status);
        }
        if burst.status == RequestStepResult::Waiting {
            bail!("scheduler is waiting but request {request_id} is still incomplete");
        }
    }

    bail!("scheduler did not complete request {request_id} before the tick limit")
}

fn cli_gpu_layers(value: i32) -> GpuLayerConfig {
    GpuLayerConfig::from_layer_count(value)
}
