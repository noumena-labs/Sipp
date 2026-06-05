//! Shared remote gateway helpers for Rust example binaries.

use std::env;

use cogentlm::{CogentEmbeddingResponse, CogentTextOptions, CogentTextResponse};
use cogentlm::{RemoteGatewayConfig, RemoteSecret};

/// Result type used by remote gateway Rust examples.
pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Command-line arguments shared by remote gateway examples.
pub struct ExampleArgs {
    pub alias: String,
    pub input: String,
}

/// Parse a gateway alias and optional input text.
pub fn args(default_input: &'static str) -> ExampleResult<ExampleArgs> {
    let mut args = env::args().skip(1);
    let alias = args.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: cargo run -p cogentlm-rust-examples --features remote --bin \
             remote_gateway_<query|chat|embed> -- <gateway-alias> [input]",
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

/// Build a remote gateway config from the alias and required environment.
pub fn gateway_remote(alias: String) -> ExampleResult<RemoteGatewayConfig> {
    Ok(RemoteGatewayConfig {
        alias,
        base_url: required_env("COGENTLM_GATEWAY_URL")?,
        token: RemoteSecret::new(required_env("COGENTLM_GATEWAY_TOKEN")?),
        timeout: None,
    })
}

/// Build text generation options from the shared example environment variables.
pub fn text_options() -> CogentTextOptions {
    CogentTextOptions {
        max_tokens: env_parse("COGENTLM_MAX_TOKENS"),
        temperature: env_parse("COGENTLM_TEMPERATURE"),
        top_p: env_parse("COGENTLM_TOP_P"),
        stop: Vec::new(),
    }
}

/// Print a remote text response.
pub fn print_text(response: CogentTextResponse) {
    println!("endpoint={:?}", response.endpoint);
    println!("finish_reason={}", response.finish_reason.as_str());
    println!("text={}", response.text.trim());
}

/// Print a compact remote embedding response preview.
pub fn print_embedding(response: CogentEmbeddingResponse) {
    let preview = response
        .values
        .iter()
        .take(8)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("endpoint={:?}", response.endpoint);
    println!("dimensions={}", response.values.len());
    println!("preview=[{preview}]");
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

fn env_parse<T>(name: &'static str) -> Option<T>
where
    T: std::str::FromStr,
{
    env::var(name).ok().and_then(|value| value.parse().ok())
}
