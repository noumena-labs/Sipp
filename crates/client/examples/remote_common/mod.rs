use std::env;

use cogentlm_client::{RemoteConfig, RemoteOpenAiConfig, RemoteSecret};

pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

pub struct ExampleArgs {
    pub model: String,
    pub input: String,
}

pub fn args(default_input: &'static str) -> ExampleResult<ExampleArgs> {
    let mut args = env::args().skip(1);
    let model = args.next().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "usage: cargo run -p cogentlm-client --example remote_<query|chat|embed> -- <remote-model> [input]",
            )
    })?;
    let input = args.collect::<Vec<_>>().join(" ");
    Ok(ExampleArgs {
        model,
        input: if input.is_empty() {
            default_input.to_string()
        } else {
            input
        },
    })
}

pub fn openai_remote(model: String) -> ExampleResult<RemoteConfig> {
    Ok(RemoteConfig::OpenAi(RemoteOpenAiConfig {
        model,
        api_key: RemoteSecret::new(required_env("OPENAI_API_KEY")?),
        base_url: env_string("COGENTLM_OPENAI_BASE_URL"),
        timeout: None,
    }))
}

fn required_env(name: &'static str) -> ExampleResult<String> {
    env_string(name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{name} is required"),
        )
        .into()
    })
}

fn env_string(name: &'static str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
