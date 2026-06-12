#![allow(dead_code)]

use std::env;
use std::path::PathBuf;

use sipp::{SippEmbeddingResponse, SippTextResponse};

pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

pub const DEFAULT_MAX_TOKENS: u32 = 2048;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_P: f32 = 0.8;
pub const DEFAULT_CONTEXT: i32 = 4096;
pub const DEFAULT_SEED: u32 = 42;

pub struct LocalArgs {
    pub model_path: PathBuf,
    pub input: String,
}

pub struct VisionArgs {
    pub model_path: PathBuf,
    pub projector_path: PathBuf,
    pub image_path: PathBuf,
    pub input: String,
}

pub struct GatewayArgs {
    pub model_path: PathBuf,
    pub target: String,
    pub input: String,
}

pub fn local_args(default_input: &'static str, command: &'static str) -> ExampleResult<LocalArgs> {
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .ok_or_else(|| usage_error(local_usage(command)))?;
    let input = args.collect::<Vec<_>>().join(" ");
    Ok(LocalArgs {
        model_path: PathBuf::from(model_path),
        input: defaulted_input(input, default_input),
    })
}

pub fn vision_args(default_input: &'static str) -> ExampleResult<VisionArgs> {
    let mut args = env::args().skip(1);
    let model_path = args.next().ok_or_else(|| {
        usage_error(
            "usage: cargo run -p sipp-rust-examples --bin vision_chat -- \
         <model.gguf> <projector.gguf> <image> [input]",
        )
    })?;
    let projector_path = args.next().ok_or_else(|| {
        usage_error(
            "usage: cargo run -p sipp-rust-examples --bin vision_chat -- \
         <model.gguf> <projector.gguf> <image> [input]",
        )
    })?;
    let image_path = args.next().ok_or_else(|| {
        usage_error(
            "usage: cargo run -p sipp-rust-examples --bin vision_chat -- \
         <model.gguf> <projector.gguf> <image> [input]",
        )
    })?;
    let input = args.collect::<Vec<_>>().join(" ");
    Ok(VisionArgs {
        model_path: PathBuf::from(model_path),
        projector_path: PathBuf::from(projector_path),
        image_path: PathBuf::from(image_path),
        input: defaulted_input(input, default_input),
    })
}

pub fn gateway_args(
    default_input: &'static str,
    command: &'static str,
) -> ExampleResult<GatewayArgs> {
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .ok_or_else(|| usage_error(gateway_usage(command)))?;
    let target = args
        .next()
        .ok_or_else(|| usage_error(gateway_usage(command)))?;
    let input = args.collect::<Vec<_>>().join(" ");
    Ok(GatewayArgs {
        model_path: PathBuf::from(model_path),
        target,
        input: defaulted_input(input, default_input),
    })
}

pub fn required_env(name: &'static str) -> ExampleResult<String> {
    env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| usage_error(format!("{name} is required")))
}

pub fn optional_env(name: &'static str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

pub fn env_parse<T>(name: &'static str) -> Option<T>
where
    T: std::str::FromStr,
{
    env::var(name).ok().and_then(|value| value.parse().ok())
}

pub fn print_text(response: SippTextResponse) {
    println!("endpoint={:?}", response.endpoint);
    println!("finish_reason={}", response.finish_reason.as_str());
    println!("text={}", response.text.trim());
    if let Some(stats) = response.local_stats {
        println!(
            "metrics=ttft_ms:{:?} decode_ms:{:.3} output_tokens:{} e2e_tps:{:?} decode_tps:{:?}",
            stats.ttft_ms,
            stats.decode_ms,
            stats.output_tokens,
            stats.e2e_tokens_per_second,
            stats.decode_tokens_per_second
        );
    }
}

pub fn print_embedding(response: SippEmbeddingResponse) {
    let preview = response
        .values
        .iter()
        .take(8)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("endpoint={:?}", response.endpoint);
    println!("dimensions={}", response.values.len());
    println!("pooling={:?}", response.pooling);
    println!("normalized={:?}", response.normalized);
    println!("preview=[{preview}]");
}

fn defaulted_input(input: String, default_input: &'static str) -> String {
    if input.is_empty() {
        default_input.to_string()
    } else {
        input
    }
}

fn local_usage(command: &'static str) -> String {
    format!("usage: cargo run -p sipp-rust-examples --bin {command} -- <model.gguf> [input]")
}

fn gateway_usage(command: &'static str) -> String {
    format!(
        "usage: cargo run -p sipp-rust-examples --features gateway --bin {command} -- \
         <model.gguf> <gateway-target> [input]"
    )
}

fn usage_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}
