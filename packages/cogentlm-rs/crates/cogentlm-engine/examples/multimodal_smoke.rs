use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use cogentlm_engine::backend::{backend_observability_json, set_llama_log_quiet};
use cogentlm_engine::chat::default_media_marker;
use cogentlm_engine::engine::{
    ChatMessage, ChatRequest, EngineEvent, EngineEventReceiver, GpuLayerConfig,
    NativeRuntimeConfig, QueryOptions, QueryRequest, RequestResult, SamplingRuntimeConfig,
};
use cogentlm_engine::lifecycle::{
    vision_model_source_from_paths, BackendPreference, ModelLoadOptions, ModelService, StatsMode,
};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    mmproj: PathBuf,
    image: PathBuf,
    prompt: String,
    max_tokens: i32,
    ctx_size: i32,
    threads: i32,
    gpu_layers: Option<i32>,
    backend: BackendPreference,
    model_store: PathBuf,
    seed: u32,
    temperature: f32,
    stats: StatsMode,
    multimodal_use_gpu: bool,
    marker_in_user_message: bool,
    raw_prompt: bool,
    verbose_llama: bool,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = env::args().skip(1);
        let Some(model) = args.next() else {
            anyhow::bail!(
                "usage: cargo run -p cogentlm-engine --example multimodal_smoke -- <model.gguf> <mmproj.gguf> <image.png> [prompt] [--max-tokens N] [--ctx-size N] [--threads N] [--gpu-layers N] [--backend auto|cpu|cuda|metal|vulkan|webgpu] [--model-store PATH] [--seed N] [--temperature F] [--stats off|basic|profile] [--cpu-mmproj] [--no-marker-in-message] [--raw] [--verbose-llama]"
            );
        };
        let Some(mmproj) = args.next() else {
            anyhow::bail!("missing <mmproj.gguf>");
        };
        let Some(image) = args.next() else {
            anyhow::bail!("missing <image.png>");
        };

        let mut out = Self {
            model: PathBuf::from(model),
            mmproj: PathBuf::from(mmproj),
            image: PathBuf::from(image),
            prompt: "Describe this image in details".to_string(),
            max_tokens: 256,
            ctx_size: 4096,
            threads: 0,
            gpu_layers: None,
            backend: BackendPreference::Auto,
            model_store: env::temp_dir().join("cogentlm-rs-model-store"),
            seed: 42,
            temperature: 0.2,
            stats: StatsMode::Basic,
            multimodal_use_gpu: true,
            marker_in_user_message: true,
            raw_prompt: false,
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
                "--stats" => {
                    let value: String = parse_next(&mut args, "--stats")?;
                    out.stats = parse_stats_mode(&value)?;
                }
                "--cpu-mmproj" => out.multimodal_use_gpu = false,
                "--no-marker-in-message" => out.marker_in_user_message = false,
                "--raw" => out.raw_prompt = true,
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

    let image = fs::read(&args.image).map_err(|error| {
        anyhow::anyhow!("failed to read image {}: {error}", args.image.display())
    })?;
    let media_marker = default_media_marker().unwrap_or_else(|_| "<image>".to_string());
    let prompt = if args.marker_in_user_message && !args.prompt.contains(&media_marker) {
        format!("{media_marker}\n{}", args.prompt)
    } else {
        args.prompt.clone()
    };

    let mut runtime = NativeRuntimeConfig::default();
    runtime.context.n_ctx = Some(args.ctx_size);
    runtime.context.n_threads = Some(args.threads);
    runtime.context.n_threads_batch = Some(args.threads);
    runtime.placement.gpu_layers = gpu_layers_config(args.gpu_layers);
    runtime.multimodal.projector_path = Some(args.mmproj.display().to_string());
    runtime.multimodal.use_gpu = Some(args.multimodal_use_gpu);
    runtime.sampling = SamplingRuntimeConfig {
        seed: Some(args.seed),
        temperature: Some(args.temperature),
        ..SamplingRuntimeConfig::default()
    };
    let load_options = ModelLoadOptions {
        backend: args.backend,
        stats: args.stats,
        runtime,
    };

    println!("multimodal_smoke");
    println!("model={}", args.model.display());
    println!("mmproj={}", args.mmproj.display());
    println!("model_store={}", args.model_store.display());
    println!("image={} bytes={}", args.image.display(), image.len());
    println!("prompt={}", args.prompt);
    println!("media_marker={media_marker}");
    println!(
        "settings=max_tokens:{} ctx:{} threads:{} gpu_layers:{:?} backend:{:?} stats:{:?} seed:{} temperature:{} multimodal_use_gpu:{} mode:{} marker_in_user_message:{}",
        args.max_tokens,
        args.ctx_size,
        args.threads,
        args.gpu_layers,
        args.backend,
        args.stats,
        args.seed,
        args.temperature,
        args.multimodal_use_gpu,
        if args.raw_prompt { "raw" } else { "chat" },
        args.marker_in_user_message
    );
    println!(
        "backend_before_load={}",
        backend_observability_json(true).unwrap_or_else(|error| format!("error:{error}"))
    );

    let load_start = Instant::now();
    let mut service = ModelService::local(&args.model_store)?;
    let loaded = service.load(
        vision_model_source_from_paths(&args.model, &args.mmproj),
        load_options,
    )?;
    let events = service.subscribe_events()?;
    println!("load_ms={:.3}", load_start.elapsed().as_secs_f64() * 1000.0);
    println!("loaded_model={:?}", loaded.model);
    println!("selected_backend={:?}", loaded.backend);
    println!("engine_state_after_load={:?}", service.state()?);
    println!(
        "backend_after_load={}",
        backend_observability_json(true).unwrap_or_else(|error| format!("error:{error}"))
    );

    let options = QueryOptions {
        context_key: "multimodal-smoke".to_string(),
        max_tokens: args.max_tokens,
        media: vec![image],
        ..QueryOptions::default()
    };
    let start = Instant::now();
    print!("\nvision_stream=");
    io::stdout().flush().ok();
    let response = if args.raw_prompt {
        service.query(
            QueryRequest::new(prompt)
                .options(options)
                .on_tokens(|batch| {
                    print!("{}", batch.text());
                    io::stdout().flush().ok();
                    Ok(())
                }),
        )
    } else {
        service.chat(
            ChatRequest::new(vec![ChatMessage::user(prompt)])
                .options(options)
                .on_tokens(|batch| {
                    print!("{}", batch.text());
                    io::stdout().flush().ok();
                    Ok(())
                }),
        )
    };

    match response {
        Ok(response) => {
            println!();
            print_response("vision", start, &response);
        }
        Err(error) => println!("\nvision_error={error}"),
    }
    println!("engine_state_after_vision={:?}", service.state()?);
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

fn parse_stats_mode(value: &str) -> anyhow::Result<StatsMode> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "off" => Ok(StatsMode::Off),
        "basic" => Ok(StatsMode::Basic),
        "profile" => Ok(StatsMode::Profile),
        _ => anyhow::bail!("--stats must be one of: off, basic, profile"),
    }
}

fn gpu_layers_config(value: Option<i32>) -> GpuLayerConfig {
    GpuLayerConfig::from_optional_layer_count(value)
}
