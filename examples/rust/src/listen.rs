mod support;

use futures::executor::block_on;
use sipp::backend::set_llama_log_quiet;
use sipp::engine::{GpuLayerConfig, NativeRuntimeConfig};
use sipp::{endpoint, SippClient, SippListenRequest};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let mut args = env::args().skip(1);
        let model_path = required_path(&mut args, "<model.gguf>")?;
        let projector_path = required_path(&mut args, "<projector.gguf>")?;
        let audio_path = required_path(&mut args, "<audio.wav|mp3|flac>")?;
        let audio = fs::read(audio_path)?;
        set_llama_log_quiet(true);

        let mut client = SippClient::new()?;
        let model = client.models().add([model_path, projector_path]).await?;
        let local = endpoint::Local::new(&model).runtime(runtime_config());
        client.add("asr", local).await?;

        let mut request = SippListenRequest::new(audio);
        if let Some(language) = support::optional_env("SIPP_LANGUAGE") {
            request = request.language(language);
        }
        let response = client.listen(request).await?;
        println!("{}", response.text.trim());
        Ok(())
    })
}

fn runtime_config() -> NativeRuntimeConfig {
    let mut config = NativeRuntimeConfig::default();
    config.placement.gpu_layers = support::env_parse("SIPP_GPU_LAYERS")
        .map(GpuLayerConfig::from_layer_count)
        .unwrap_or(GpuLayerConfig::Auto);
    config.context.n_ctx = support::env_parse("SIPP_CONTEXT").or(Some(8192));
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
            format!("usage: listen <model.gguf> <projector.gguf> <audio>; missing {name}"),
        )
        .into()
    })
}
