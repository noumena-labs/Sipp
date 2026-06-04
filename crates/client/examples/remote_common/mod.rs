use std::env;

use cogentlm_client::{RemoteGatewayConfig, RemoteSecret};

pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

pub struct ExampleArgs {
    pub alias: String,
    pub input: String,
}

pub fn args(default_input: &'static str) -> ExampleResult<ExampleArgs> {
    let mut args = env::args().skip(1);
    let alias = args.next().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "usage: cargo run -p cogentlm-client --example remote_gateway_<query|chat|embed> -- <gateway-alias> [input]",
            )
    })?;
    let input = args.collect::<Vec<_>>().join(" ");
    Ok(ExampleArgs {
        alias,
        input: if input.is_empty() {
            default_input.to_string()
        } else {
            input
        },
    })
}

pub fn gateway_remote(alias: String) -> ExampleResult<RemoteGatewayConfig> {
    Ok(RemoteGatewayConfig {
        alias,
        base_url: required_env("COGENTLM_GATEWAY_URL")?,
        token: RemoteSecret::new(required_env("COGENTLM_GATEWAY_TOKEN")?),
        timeout: None,
    })
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
    env::var(name).ok().filter(|value| !value.is_empty())
}
