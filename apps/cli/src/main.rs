#![allow(
    clippy::empty_line_after_doc_comments,
    reason = "test and source section banners follow repository style"
)]

use std::fs;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context};
use clap::{Parser, ValueEnum};
use serde_json::json;
use sipp::backend::set_llama_log_quiet;
use sipp::engine::{GpuLayerConfig, NativeRuntimeConfig, SamplingRuntimeConfig};
use sipp::lifecycle::{BackendPolicy, BackendPreference, ModelLoadOptions, StatsMode};
use sipp::runtime::metrics::RuntimeObservabilityMetrics;
use sipp::runtime::request::{GenerateResponseStatus, ResponseOutput};
use sipp::runtime::{InferenceRuntime, RequestStepResult};
use sipp::{endpoint, SippClient, SippListenRequest, SippSpeakRequest};

const DEFAULT_TEXT_MAX_TOKENS: u32 = 64;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/main_tests.rs"]
mod main_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Parser)]
#[command(name = "sipp")]
#[command(about = "Sipp Rust runtime proof-of-concept CLI")]
struct Args {
    /// Path to a GGUF model.
    model: PathBuf,

    /// Prompt text for generation or speech synthesis.
    prompt: Option<String>,

    /// Audio projector paired with an ASR or TTS model.
    #[arg(long)]
    projector: Option<PathBuf>,

    /// Transcribe encoded WAV, MP3, or FLAC audio.
    #[arg(long, conflicts_with = "speak")]
    listen: Option<PathBuf>,

    /// Synthesize the prompt and write a mono PCM16 WAV file.
    #[arg(long, value_name = "OUTPUT_WAV", conflicts_with = "listen")]
    speak: Option<PathBuf>,

    /// Optional ASR or TTS language hint.
    #[arg(long)]
    language: Option<String>,

    /// Optional encoded speaker-reference audio for TTS.
    #[arg(long, requires = "speak")]
    speaker: Option<PathBuf>,

    /// Maximum synthesized duration in milliseconds.
    #[arg(long, value_name = "MILLISECONDS", requires = "speak")]
    max_duration_ms: Option<NonZeroU32>,

    /// Maximum generated tokens. Text defaults to 64; listen uses the core default.
    #[arg(long)]
    max_tokens: Option<NonZeroU32>,

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
    validate_args(&args)?;
    set_llama_log_quiet(true);

    let mut stdout = io::stdout().lock();
    run_native_runtime(&args, &mut stdout)
        .with_context(|| format!("native runtime failed for {}", args.model.display()))?;

    if args.speak.is_none() {
        writeln!(stdout)?;
    }
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    let speech_mode = args.listen.is_some() || args.speak.is_some();
    if !speech_mode && args.prompt.is_none() {
        bail!("prompt is required unless --listen is used");
    }
    if speech_mode && args.projector.is_none() {
        bail!("--projector is required for --listen and --speak");
    }
    if args.speak.is_some() && args.prompt.is_none() {
        bail!("prompt is required for --speak");
    }
    if let Some(language) = &args.language {
        if language.trim().is_empty() || language.trim() != language {
            bail!("--language must not be empty or contain surrounding whitespace");
        }
    }
    Ok(())
}

fn run_native_runtime(args: &Args, stdout: &mut impl Write) -> anyhow::Result<()> {
    let load_options = ModelLoadOptions {
        backend: args.backend.to_preference(),
        stats: args.stats.to_lifecycle_stats(),
        runtime: runtime_config_from_args(args),
    };
    let backend_plan = BackendPolicy::select(&load_options)?;
    if args.listen.is_some() || args.speak.is_some() {
        return run_speech(args, backend_plan.config, stdout);
    }
    let mut runtime = InferenceRuntime::load(&args.model, backend_plan.config)?;
    let prompt = args
        .prompt
        .as_deref()
        .context("prompt is required unless --listen is used")?;
    let prompt = if args.chat {
        let messages = json!([{ "role": "user", "content": prompt }]);
        let rendered = runtime.apply_chat_template_json(&messages.to_string(), true)?;
        if rendered.is_empty() {
            bail!("model did not provide a usable chat template");
        }
        rendered
    } else {
        prompt.to_string()
    };

    let request_id = runtime.enqueue_request(
        "",
        prompt,
        args.max_tokens
            .map(NonZeroU32::get)
            .unwrap_or(DEFAULT_TEXT_MAX_TOKENS)
            .min(i32::MAX as u32) as i32,
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
                    ResponseOutput::Audio(_) => {
                        bail!("generation request completed with audio output")
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

fn run_speech(
    args: &Args,
    config: NativeRuntimeConfig,
    stdout: &mut impl Write,
) -> anyhow::Result<()> {
    let projector = args
        .projector
        .as_ref()
        .context("--projector is required for --listen and --speak")?;
    let mut config = config;
    config.multimodal.projector_path = Some(projector.to_string_lossy().into_owned());

    let mut client = SippClient::new()?;
    let model =
        futures::executor::block_on(client.models().add([args.model.clone(), projector.clone()]))?;
    let local = endpoint::Local::new(&model).runtime(config);
    futures::executor::block_on(client.add("speech", local))?;

    if let Some(audio_path) = &args.listen {
        let audio = fs::read(audio_path)
            .with_context(|| format!("failed to read {}", audio_path.display()))?;
        let response = futures::executor::block_on(client.listen(SippListenRequest {
            endpoint: None,
            audio,
            language: args.language.clone(),
            max_tokens: args.max_tokens.map(NonZeroU32::get),
        }))?;
        stdout.write_all(response.text.as_bytes())?;
        Ok(())
    } else {
        let output_path = args.speak.as_ref().context("speech mode is missing")?;
        let text = args
            .prompt
            .as_ref()
            .context("prompt is required for --speak")?;
        let speaker_audio = args
            .speaker
            .as_ref()
            .map(|path| {
                fs::read(path).with_context(|| format!("failed to read {}", path.display()))
            })
            .transpose()?;
        let response = futures::executor::block_on(client.speak(SippSpeakRequest {
            endpoint: None,
            text: text.clone(),
            language: args.language.clone(),
            speaker_audio,
            max_duration_ms: args.max_duration_ms.map(NonZeroU32::get),
        }))?;
        fs::write(output_path, &response.audio)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        eprintln!(
            "wrote {} ms at {} Hz to {}",
            response.duration_ms,
            response.sample_rate_hz,
            output_path.display()
        );
        Ok(())
    }
}

fn runtime_config_from_args(args: &Args) -> NativeRuntimeConfig {
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
    config.multimodal.projector_path = args
        .projector
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned());
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
    config
}

fn print_stats(mode: CliStatsMode, stats: RuntimeObservabilityMetrics) -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    print_stats_to_writer(mode, stats, &mut stderr)
}

fn print_stats_to_writer(
    mode: CliStatsMode,
    stats: RuntimeObservabilityMetrics,
    writer: &mut impl Write,
) -> io::Result<()> {
    if mode == CliStatsMode::Off {
        return Ok(());
    }

    writeln!(writer)?;
    writeln!(writer, "stats:")?;
    writeln!(writer, "  input_tokens: {}", stats.input_tokens)?;
    writeln!(writer, "  output_tokens: {}", stats.output_tokens)?;
    writeln!(writer, "  prefill_tokens: {}", stats.prefill_tokens)?;
    writeln!(writer, "  cache_hits: {}", stats.cache_hits)?;
    write_optional_ms(writer, "ttft_ms", stats.ttft_ms)?;
    write_optional_ms(writer, "inter_token_ms", stats.itl_avg_ms)?;
    write_optional_ms(writer, "e2e_ms", stats.e2e_ms)?;
    write_optional_ms(writer, "prefill_ms", stats.prefill_ms)?;
    write_optional_ms(writer, "decode_ms", stats.decode_ms)?;
    write_token_rate(
        writer,
        "e2e_tokens_per_second",
        stats.output_tokens,
        stats.e2e_ms,
    )?;
    write_token_rate(
        writer,
        "decode_tokens_per_second",
        stats.output_tokens,
        stats.decode_ms,
    )?;

    if mode == CliStatsMode::Profile {
        write_optional_ms(writer, "backend_ms", stats.native_gpu_ms)?;
        write_optional_ms(writer, "sync_ms", stats.native_sync_ms)?;
        write_optional_ms(writer, "engine_overhead_ms", stats.native_logic_ms)?;
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
