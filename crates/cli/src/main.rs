use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context};
use clap::{Parser, ValueEnum};
use cogentlm_engine::backend::set_llama_log_quiet;
use cogentlm_engine::engine::{GpuLayerConfig, NativeRuntimeConfig, SamplingRuntimeConfig};
use cogentlm_engine::lifecycle::{BackendPolicy, BackendPreference, ModelLoadOptions, StatsMode};
use cogentlm_engine::runtime::metrics::RuntimeObservabilityMetrics;
use cogentlm_engine::runtime::request::{GenerateResponseStatus, ResponseOutput};
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
    #[arg(long, default_value_t = 8196)]
    ctx_size: u32,

    /// Decode batch size in tokens.
    #[arg(long, default_value_t = 512)]
    batch_size: u32,

    /// Number of model layers to offload to GPU.
    #[arg(long, allow_negative_numbers = true)]
    gpu_layers: Option<i32>,

    /// Backend preference for model execution.
    #[arg(long, value_enum, default_value_t = CliBackend::Auto)]
    backend: CliBackend,

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

    /// Print request timing/token stats to stderr.
    #[arg(long, value_enum, default_value_t = CliStatsMode::Off)]
    stats: CliStatsMode,

    /// Render the prompt as a single user chat message before generation.
    #[arg(long)]
    chat: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
enum CliStatsMode {
    #[default]
    Off,
    Basic,
    Profile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
enum CliBackend {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

impl CliBackend {
    fn to_preference(self) -> BackendPreference {
        match self {
            Self::Auto => BackendPreference::Auto,
            Self::Cpu => BackendPreference::Cpu,
            Self::Cuda => BackendPreference::Cuda,
            Self::Metal => BackendPreference::Metal,
            Self::Vulkan => BackendPreference::Vulkan,
        }
    }
}

impl CliStatsMode {
    fn to_lifecycle_stats(self) -> StatsMode {
        match self {
            Self::Off => StatsMode::Off,
            Self::Basic => StatsMode::Basic,
            Self::Profile => StatsMode::Profile,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    set_llama_log_quiet(true);

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
    if let Some(gpu_layers) = args.gpu_layers {
        config.placement.gpu_layers = GpuLayerConfig::from_layer_count(gpu_layers);
    }
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

    let load_options = ModelLoadOptions {
        backend: args.backend.to_preference(),
        stats: args.stats.to_lifecycle_stats(),
        runtime: config,
    };
    let backend_plan = BackendPolicy::select(&load_options)?;
    let mut runtime = InferenceRuntime::load(&args.model, backend_plan.config)?;
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
        false,
    )?;

    for _ in 0..10_000 {
        let burst = runtime.run_scheduler_burst(256, 1, 0, Duration::ZERO);
        if let Some(response) = runtime.take_completed_response(request_id) {
            if response.status == GenerateResponseStatus::Completed {
                let output = match response.output {
                    ResponseOutput::Text(text) => text,
                    ResponseOutput::Embedding { .. } => {
                        bail!("generation request completed with embedding output")
                    }
                };
                stdout.write_all(output.as_bytes())?;
                print_stats(args.stats, response.runtime_observability)?;
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

fn print_stats(mode: CliStatsMode, stats: RuntimeObservabilityMetrics) -> io::Result<()> {
    if mode == CliStatsMode::Off {
        return Ok(());
    }

    let mut stderr = io::stderr().lock();
    writeln!(stderr)?;
    writeln!(stderr, "stats:")?;
    writeln!(stderr, "  input_tokens: {}", stats.input_tokens)?;
    writeln!(stderr, "  output_tokens: {}", stats.output_tokens)?;
    writeln!(stderr, "  prefill_tokens: {}", stats.prefill_tokens)?;
    writeln!(stderr, "  cache_hits: {}", stats.cache_hits)?;
    write_optional_ms(&mut stderr, "ttft_ms", stats.ttft_ms)?;
    write_optional_ms(&mut stderr, "inter_token_ms", stats.itl_avg_ms)?;
    write_optional_ms(&mut stderr, "e2e_ms", stats.e2e_ms)?;
    write_optional_ms(&mut stderr, "prefill_ms", stats.prefill_ms)?;
    write_optional_ms(&mut stderr, "decode_ms", stats.decode_ms)?;
    write_token_rate(
        &mut stderr,
        "e2e_tokens_per_second",
        stats.output_tokens,
        stats.e2e_ms,
    )?;
    write_token_rate(
        &mut stderr,
        "decode_tokens_per_second",
        stats.output_tokens,
        stats.decode_ms,
    )?;

    if mode == CliStatsMode::Profile {
        write_optional_ms(&mut stderr, "backend_ms", stats.native_gpu_ms)?;
        write_optional_ms(&mut stderr, "sync_ms", stats.native_sync_ms)?;
        write_optional_ms(&mut stderr, "engine_overhead_ms", stats.native_logic_ms)?;
    }

    Ok(())
}

fn write_optional_ms(writer: &mut impl Write, label: &str, value: f64) -> io::Result<()> {
    if value > 0.0 {
        writeln!(writer, "  {label}: {value:.2}")?;
    }
    Ok(())
}

fn write_token_rate(
    writer: &mut impl Write,
    label: &str,
    tokens: i32,
    elapsed_ms: f64,
) -> io::Result<()> {
    if tokens > 0 && elapsed_ms > 0.0 {
        let value = f64::from(tokens) / (elapsed_ms / 1000.0);
        writeln!(writer, "  {label}: {value:.2}")?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/main_tests.rs"]
mod main_tests;
