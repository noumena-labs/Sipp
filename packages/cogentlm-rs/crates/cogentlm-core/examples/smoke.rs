use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cogentlm_core::{
    backend_observability_json, model_source_from_path, set_llama_log_quiet, BackendPreference,
    ChatMessage, ChatRequest, EngineEvent, EngineEventReceiver, FlashAttentionMode, GpuLayerConfig,
    ModelLoadOptions, ModelService, NativeRuntimeConfig, QueryOptions, RequestResult,
    SamplingRuntimeConfig, StatsMode,
};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    prompt: String,
    max_tokens: i32,
    ctx_size: i32,
    threads: i32,
    gpu_layers: Option<i32>,
    backend: BackendPreference,
    model_store: PathBuf,
    seed: u32,
    temperature: f32,
    top_k: Option<i32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    flash_attention: FlashAttentionMode,
    fit_params: bool,
    stats: StatsMode,
    stream_mode: StreamMode,
    verbose_llama: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Off,
    Silent,
    Print,
}

impl StreamMode {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "silent" => Ok(Self::Silent),
            "print" => Ok(Self::Print),
            _ => anyhow::bail!("--stream must be one of: off, silent, print"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Silent => "silent",
            Self::Print => "print",
        }
    }
}

#[derive(Debug, Default)]
struct StreamCapture {
    frames: u64,
    bytes: u64,
    batches: u64,
}

impl StreamCapture {
    fn record(&mut self, batch: &cogentlm_core::TokenBatch) {
        self.frames = self.frames.saturating_add(u64::from(batch.frame_count));
        self.bytes = self.bytes.saturating_add(u64::from(batch.byte_count));
        self.batches = self.batches.saturating_add(1);
    }
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = env::args().skip(1);
        let Some(model) = args.next() else {
            anyhow::bail!(
                "usage: cargo run -p cogentlm-core --example phase3_smoke -- <model.gguf> [prompt] [--max-tokens N] [--ctx-size N] [--threads N] [--gpu-layers N] [--backend auto|cpu|cuda|metal|vulkan|webgpu] [--model-store PATH] [--seed N] [--temperature F] [--top-k N] [--top-p F] [--min-p F] [--repeat-penalty F] [--flash-attn auto|on|off] [--fit on|off] [--stats off|basic|profile] [--stream off|silent|print] [--verbose-llama]"
            );
        };

        let mut out = Self {
            model: PathBuf::from(model),
            prompt: "Describe browser LLM inference.".to_string(),
            max_tokens: 4096,
            ctx_size: 8192,
            threads: 0,
            gpu_layers: None,
            backend: BackendPreference::Auto,
            model_store: env::temp_dir().join("cogentlm-rs-model-store"),
            seed: 42,
            temperature: 0.7,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: None,
            flash_attention: FlashAttentionMode::Auto,
            fit_params: false,
            stats: StatsMode::Basic,
            stream_mode: StreamMode::Off,
            verbose_llama: false,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--max-tokens" => out.max_tokens = parse_next(&mut args, "--max-tokens")?,
                "--ctx-size" => out.ctx_size = parse_next(&mut args, "--ctx-size")?,
                "--threads" => out.threads = parse_next(&mut args, "--threads")?,
                "--gpu-layers" => out.gpu_layers = Some(parse_next(&mut args, "--gpu-layers")?),
                "--backend" => {
                    let value: String = parse_next(&mut args, "--backend")?;
                    out.backend = parse_backend(&value)?;
                }
                "--model-store" => out.model_store = parse_next(&mut args, "--model-store")?,
                "--seed" => out.seed = parse_next(&mut args, "--seed")?,
                "--temperature" => out.temperature = parse_next(&mut args, "--temperature")?,
                "--top-k" => out.top_k = Some(parse_next(&mut args, "--top-k")?),
                "--top-p" => out.top_p = Some(parse_next(&mut args, "--top-p")?),
                "--min-p" => out.min_p = Some(parse_next(&mut args, "--min-p")?),
                "--repeat-penalty" => {
                    out.repeat_penalty = Some(parse_next(&mut args, "--repeat-penalty")?)
                }
                "--flash-attn" => {
                    let value: String = parse_next(&mut args, "--flash-attn")?;
                    out.flash_attention = parse_flash_attention(&value)?;
                }
                "--fit" => {
                    let value: String = parse_next(&mut args, "--fit")?;
                    out.fit_params = parse_bool_flag(&value, "--fit")?;
                }
                "--stats" => {
                    let value: String = parse_next(&mut args, "--stats")?;
                    out.stats = parse_stats_mode(&value)?;
                }
                "--stream" => {
                    let value: String = parse_next(&mut args, "--stream")?;
                    out.stream_mode = StreamMode::parse(&value)?;
                }
                "--stream-silent" => out.stream_mode = StreamMode::Silent,
                "--stream-print" => out.stream_mode = StreamMode::Print,
                "--verbose-llama" => out.verbose_llama = true,
                value if value.starts_with("--") => anyhow::bail!("unknown option: {value}"),
                value => out.prompt = value.to_string(),
            }
        }

        Ok(out)
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    if !args.verbose_llama {
        set_llama_log_quiet(true);
    }

    let mut runtime = NativeRuntimeConfig::default();
    runtime.context.n_ctx = Some(args.ctx_size);
    runtime.context.n_threads = Some(args.threads);
    runtime.context.n_threads_batch = Some(args.threads);
    runtime.context.flash_attention = args.flash_attention;
    runtime.placement.gpu_layers = gpu_layers_config(args.gpu_layers);
    runtime.placement.fit_params = args.fit_params;
    runtime.sampling = SamplingRuntimeConfig {
        seed: Some(args.seed),
        temperature: Some(args.temperature),
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        repeat_penalty: args.repeat_penalty,
        ..SamplingRuntimeConfig::default()
    };
    let load_options = ModelLoadOptions {
        backend: args.backend,
        stats: args.stats,
        runtime,
    };

    println!("phase3_smoke");
    println!("model={}", args.model.display());
    println!("model_store={}", args.model_store.display());
    println!("prompt={}", args.prompt);
    println!(
        "settings=max_tokens:{} ctx:{} threads:{} gpu_layers:{:?} backend:{:?} stats:{:?} seed:{} temperature:{} top_k:{:?} top_p:{:?} min_p:{:?} repeat_penalty:{:?} flash_attn:{:?} fit:{} stream:{}",
        args.max_tokens,
        args.ctx_size,
        args.threads,
        args.gpu_layers,
        args.backend,
        args.stats,
        args.seed,
        args.temperature,
        args.top_k,
        args.top_p,
        args.min_p,
        args.repeat_penalty,
        args.flash_attention,
        args.fit_params,
        args.stream_mode.as_str()
    );
    println!(
        "backend_before_load={}",
        backend_observability_json(true).unwrap_or_else(|error| format!("error:{error}"))
    );

    let load_start = Instant::now();
    let mut service = ModelService::local(&args.model_store)?;
    let loaded = service.load(model_source_from_path(&args.model), load_options)?;
    let events = service.subscribe_events()?;
    println!("load_ms={:.3}", load_start.elapsed().as_secs_f64() * 1000.0);
    println!("loaded_model={:?}", loaded.model);
    println!("selected_backend={:?}", loaded.backend);
    println!("engine_state_after_load={:?}", service.state()?);
    println!(
        "backend_after_load={}",
        backend_observability_json(true).unwrap_or_else(|error| format!("error:{error}"))
    );

    // --- Chat ---
    let chat_options = QueryOptions {
        context_key: "phase3-smoke-chat".to_string(),
        max_tokens: args.max_tokens,
        ..QueryOptions::default()
    };
    let chat_start = Instant::now();
    let stream_capture = Arc::new(Mutex::new(StreamCapture::default()));
    let mut chat_request =
        ChatRequest::new(vec![ChatMessage::user(&args.prompt)]).options(chat_options);
    match args.stream_mode {
        StreamMode::Off => {}
        StreamMode::Silent => {
            let stream_capture = Arc::clone(&stream_capture);
            chat_request = chat_request.on_tokens(move |batch| {
                if let Ok(mut capture) = stream_capture.lock() {
                    capture.record(batch);
                }
                Ok(())
            });
        }
        StreamMode::Print => {
            print!("\nchat_stream=");
            io::stdout().flush().ok();
            let stream_capture = Arc::clone(&stream_capture);
            chat_request = chat_request.on_tokens(move |batch| {
                print!("{}", batch.text());
                if let Ok(mut capture) = stream_capture.lock() {
                    capture.record(batch);
                }
                Ok(())
            });
        }
    }
    match service.chat(chat_request) {
        Ok(chat) => {
            if args.stream_mode == StreamMode::Print {
                io::stdout().flush().ok();
                println!();
            }
            print_response("chat", chat_start, &chat);
            if args.stream_mode != StreamMode::Off {
                if let Ok(capture) = stream_capture.lock() {
                    println!(
                        "chat_stream_stats=batches:{} frames:{} bytes:{}",
                        capture.batches, capture.frames, capture.bytes
                    );
                }
            }
        }
        Err(error) => println!("\nchat_error={error}"),
    }
    println!("engine_state_after_chat={:?}", service.state()?);
    print_event_summary(&events);

    service.close()?;
    Ok(())
}

fn print_response(label: &str, start: Instant, response: &RequestResult) {
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let stats = response.stats;

    println!("{label}_finish_reason={:?}", response.finish_reason);
    println!("{label}_wall_ms={wall_ms:.3}");
    println!(
        "{label}_metrics=ttft_ms:{:?} inter_token_ms:{:?} e2e_ms:{:?} prefill_ms:{:.3} decode_ms:{:.3} input_tokens:{} output_tokens:{} cache_hits:{} request_tps_e2e:{:?} decode_tps:{:?}",
        stats.ttft_ms,
        stats.inter_token_ms,
        stats.e2e_ms,
        stats.prefill_ms,
        stats.decode_ms,
        stats.input_tokens,
        stats.output_tokens,
        stats.cache_hits,
        stats.tokens_per_second,
        stats.decode_tokens_per_second
    );
    println!(
        "{label}_debug_metrics_counts=scheduler_ticks:{} decode_ticks:{} prefill_ticks:{} backend_sampler_attach_attempts:{} backend_sampler_attach_failures:{}",
        stats.debug_metrics_scheduler_ticks,
        stats.debug_metrics_decode_ticks,
        stats.debug_metrics_prefill_ticks,
        stats.debug_metrics_backend_sampler_attach_attempts,
        stats.debug_metrics_backend_sampler_attach_failures
    );
    println!(
        "{label}_debug_metrics_stage_ms=admit:{:.3} normalize:{:.3} backend_sampler_attach:{:.3} select_slots:{:.3} plan:{:.3} batch_build:{:.3} llama_decode:{:.3} llama_sync:{:.3} apply_bookkeeping:{:.3} apply_decode_results:{:.3} sample:{:.3} token_piece:{:.3} emit:{:.3} prefix_queue:{:.3} finalize:{:.3} commit_observability:{:.3} post_decode:{:.3}",
        stats.debug_metrics_admit_ms,
        stats.debug_metrics_normalize_ms,
        stats.debug_metrics_backend_sampler_attach_ms,
        stats.debug_metrics_select_slots_ms,
        stats.debug_metrics_plan_ms,
        stats.debug_metrics_batch_build_ms,
        stats.debug_metrics_llama_decode_ms,
        stats.debug_metrics_llama_sync_ms,
        stats.debug_metrics_apply_bookkeeping_ms,
        stats.debug_metrics_apply_decode_results_ms,
        stats.debug_metrics_sample_ms,
        stats.debug_metrics_token_piece_ms,
        stats.debug_metrics_emit_ms,
        stats.debug_metrics_prefix_queue_ms,
        stats.debug_metrics_finalize_ms,
        stats.debug_metrics_commit_observability_ms,
        stats.debug_metrics_post_decode_ms
    );
}

fn print_event_summary(events: &EngineEventReceiver) {
    let mut state_events = 0;
    let mut request_started = 0;
    let mut request_completed = 0;
    let mut request_failed = 0;
    for event in events.try_iter() {
        match event {
            EngineEvent::State(_) => state_events += 1,
            EngineEvent::RequestStarted { .. } => request_started += 1,
            EngineEvent::RequestCompleted { .. } => request_completed += 1,
            EngineEvent::RequestFailed { .. } => request_failed += 1,
            EngineEvent::LoadProgress { .. } | EngineEvent::Closed => {}
        }
    }
    println!(
        "engine_events=state:{} request_started:{} request_completed:{} request_failed:{}",
        state_events, request_started, request_completed, request_failed
    );
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> anyhow::Result<T>
where
    T::Err: std::fmt::Display,
{
    let Some(value) = args.next() else {
        anyhow::bail!("{flag} requires a value");
    };
    value
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid value for {flag}: {error}"))
}

fn parse_backend(value: &str) -> anyhow::Result<BackendPreference> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "auto" => Ok(BackendPreference::Auto),
        "cpu" => Ok(BackendPreference::Cpu),
        "cuda" => Ok(BackendPreference::Cuda),
        "metal" => Ok(BackendPreference::Metal),
        "vulkan" => Ok(BackendPreference::Vulkan),
        "webgpu" | "web_gpu" => Ok(BackendPreference::WebGpu),
        _ => anyhow::bail!("--backend must be one of: auto, cpu, cuda, metal, vulkan, webgpu"),
    }
}

fn parse_flash_attention(value: &str) -> anyhow::Result<FlashAttentionMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "auto" => Ok(FlashAttentionMode::Auto),
        "on" | "true" | "enabled" => Ok(FlashAttentionMode::Enabled),
        "off" | "false" | "disabled" => Ok(FlashAttentionMode::Disabled),
        _ => anyhow::bail!("--flash-attn must be one of: auto, on, off"),
    }
}

fn parse_bool_flag(value: &str, flag: &'static str) -> anyhow::Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => anyhow::bail!("{flag} must be one of: on, off"),
    }
}

fn parse_stats_mode(value: &str) -> anyhow::Result<StatsMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "off" => Ok(StatsMode::Off),
        "basic" => Ok(StatsMode::Basic),
        "profile" => Ok(StatsMode::Profile),
        _ => anyhow::bail!("--stats must be one of: off, basic, profile"),
    }
}

fn gpu_layers_config(value: Option<i32>) -> GpuLayerConfig {
    match value {
        None => GpuLayerConfig::Auto,
        Some(-2) => GpuLayerConfig::All,
        Some(count) => GpuLayerConfig::Count(count),
    }
}
