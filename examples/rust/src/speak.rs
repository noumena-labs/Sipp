mod support;

use futures::executor::block_on;
use sipp::backend::set_llama_log_quiet;
use sipp::engine::{GpuLayerConfig, NativeRuntimeConfig};
use sipp::{endpoint, SippClient, SippSpeakRequest};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let mut args = env::args().skip(1);
        let model_path = required_path(&mut args, "<model.gguf>")?;
        let projector_path = required_path(&mut args, "<projector.gguf>")?;
        let output_path = required_path(&mut args, "<output.wav>")?;
        let text = args.collect::<Vec<_>>().join(" ");
        let text = if text.is_empty() {
            "Hello from Sipp."
        } else {
            &text
        };
        set_llama_log_quiet(true);

        let mut client = SippClient::new()?;
        let model = client.models().add([model_path, projector_path]).await?;
        let local = endpoint::Local::new(&model).runtime(runtime_config());
        client.add("tts", local).await?;

        let mut request = SippSpeakRequest::new(text);
        if let Some(language) = support::optional_env("SIPP_LANGUAGE") {
            request = request.language(language);
        }
        if let Some(speaker_path) = support::optional_env("SIPP_SPEAKER_AUDIO") {
            request = request.speaker(fs::read(speaker_path)?);
        }
        if let Some(max_duration_ms) = support::optional_env("SIPP_MAX_DURATION_MS") {
            request = request.max_duration_ms(max_duration_ms.parse()?);
        }
        let response = client.speak(request).await?;
        fs::write(&output_path, response.audio)?;
        println!(
            "wrote {} ms at {} Hz to {}",
            response.duration_ms,
            response.sample_rate_hz,
            output_path.display()
        );
        Ok(())
    })
}

fn runtime_config() -> NativeRuntimeConfig {
    let mut config = NativeRuntimeConfig::default();
    config.placement.gpu_layers = support::env_parse("SIPP_GPU_LAYERS")
        .map(GpuLayerConfig::from_layer_count)
        .unwrap_or(GpuLayerConfig::Auto);
    config.context.n_ctx = support::env_parse("SIPP_CONTEXT").or(Some(4096));
    config.context.n_threads = support::env_parse("SIPP_THREADS");
    config.context.n_threads_batch = support::env_parse("SIPP_THREADS");
    config.context.warmup = false;
    config
}

fn required_path(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> support::ExampleResult<PathBuf> {
    args.next().map(PathBuf::from).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "usage: speak <model.gguf> <projector.gguf> <output.wav> [text]; missing {name}"
            ),
        )
        .into()
    })
}
