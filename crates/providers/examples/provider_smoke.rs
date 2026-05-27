use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Parser, ValueEnum};
use cogentlm_core::{ChatMessage, ChatRole};
use cogentlm_providers::{
    AnthropicConfig, OpenAiConfig, ProviderAuth, ProviderChatRequest, ProviderClient,
    ProviderGenerationOptions, ProviderOptions, ProviderStreamEvent, ProxyConfig, ProxyProtocol,
    SecretString,
};
use futures_util::StreamExt;

#[derive(Debug, Parser)]
#[command(
    name = "provider_smoke",
    about = "Smoke-test CogentLM provider chat APIs"
)]
struct Args {
    /// Provider preset to use. Falls back to COGENT_PROVIDER, then gemini.
    #[arg(long, value_enum)]
    provider: Option<ProviderPreset>,

    /// Provider model id. Falls back to COGENT_MODEL.
    #[arg(long)]
    model: Option<String>,

    /// Prompt to send for chat/stream tests. Falls back to COGENT_PROMPT.
    #[arg(long)]
    prompt: Option<String>,

    /// Override provider API prefix. Falls back to COGENT_BASE_URL.
    #[arg(long)]
    base_url: Option<String>,

    /// API key value. Prefer --api-key-env or .env for normal use.
    #[arg(long)]
    api_key: Option<String>,

    /// Environment variable containing the API key. Falls back to COGENT_API_KEY_ENV.
    #[arg(long)]
    api_key_env: Option<String>,

    /// Load environment variables from this file before resolving the API key.
    #[arg(long, default_value = ".env")]
    env_file: PathBuf,

    /// Do not load a .env file.
    #[arg(long)]
    no_env_file: bool,

    /// List models and exit. Also enabled by COGENT_LIST_MODELS=true.
    #[arg(long)]
    list_models: bool,

    /// Stream chat output instead of waiting for a full response. Also enabled by COGENT_STREAM=true.
    #[arg(long)]
    stream: bool,

    /// Request timeout in milliseconds. Falls back to COGENT_TIMEOUT_MS.
    #[arg(long)]
    timeout_ms: Option<u64>,

    /// Extra static header for proxy providers, formatted as name=value. Repeatable.
    #[arg(long = "header")]
    headers: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProviderPreset {
    Openai,
    Anthropic,
    Gemini,
    Deepseek,
    Qwen,
    Kimi,
    Glm,
    Proxy,
}

impl ProviderPreset {
    fn default_base_url(self) -> Option<&'static str> {
        match self {
            Self::Openai => None,
            Self::Anthropic => None,
            Self::Gemini => Some("https://generativelanguage.googleapis.com/v1beta/openai/"),
            Self::Deepseek => Some("https://api.deepseek.com"),
            Self::Qwen => Some("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            Self::Kimi => Some("https://api.moonshot.ai/v1"),
            Self::Glm => Some("https://open.bigmodel.cn/api/paas/v4"),
            Self::Proxy => None,
        }
    }

    fn default_api_key_env(self) -> &'static str {
        match self {
            Self::Openai => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::Gemini => "GEMINI_API_KEY",
            Self::Deepseek => "DEEPSEEK_API_KEY",
            Self::Qwen => "DASHSCOPE_API_KEY",
            Self::Kimi => "MOONSHOT_API_KEY",
            Self::Glm => "ZAI_API_KEY",
            Self::Proxy => "PROVIDER_API_KEY",
        }
    }

    fn default_model(self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("claude-sonnet-4-20250514"),
            Self::Gemini => Some("gemini-2.5-flash"),
            _ => None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if !args.no_env_file {
        load_env_file(&args.env_file)?;
    }

    let provider = resolve_provider(&args)?;
    let client = build_client(&args, provider)?;

    if args.list_models || env_bool("COGENT_LIST_MODELS")?.unwrap_or(false) {
        for model in client.list_models().await? {
            println!("{}", model.id);
        }
        return Ok(());
    }

    let model = args
        .model
        .clone()
        .or_else(|| env_string("COGENT_MODEL"))
        .or_else(|| provider.default_model().map(str::to_owned))
        .ok_or_else(|| {
            format!(
                "--model is required for provider {:?}; use --list-models to inspect available models",
                provider
            )
        })?;
    let prompt = args
        .prompt
        .clone()
        .or_else(|| env_string("COGENT_PROMPT"))
        .unwrap_or_else(|| "Say hi in one sentence.".to_string());

    let request = ProviderChatRequest {
        model,
        messages: vec![ChatMessage::new(ChatRole::User, prompt)],
        options: ProviderGenerationOptions::default(),
        provider_options: ProviderOptions::new(),
    };

    if args.stream || env_bool("COGENT_STREAM")?.unwrap_or(false) {
        stream_chat(&client, request).await?;
    } else {
        let response = client.chat(request).await?;
        println!("{}", response.result.text);
        if let Some(usage) = response.usage {
            eprintln!(
                "usage input={:?} output={:?} total={:?}",
                usage.input_tokens, usage.output_tokens, usage.total_tokens
            );
        }
    }

    Ok(())
}

fn build_client(
    args: &Args,
    provider: ProviderPreset,
) -> Result<ProviderClient, Box<dyn std::error::Error>> {
    let api_key = resolve_api_key(args, provider)?;
    let timeout_ms = match args.timeout_ms {
        Some(timeout_ms) => Some(timeout_ms),
        None => env_u64("COGENT_TIMEOUT_MS")?,
    };
    let timeout = timeout_ms.map(Duration::from_millis);

    match provider {
        ProviderPreset::Openai => Ok(ProviderClient::openai(OpenAiConfig {
            api_key: SecretString::new(api_key),
            base_url: args
                .base_url
                .clone()
                .or_else(|| env_string("COGENT_BASE_URL")),
            timeout,
        })?),
        ProviderPreset::Anthropic => Ok(ProviderClient::anthropic(AnthropicConfig {
            api_key: SecretString::new(api_key),
            base_url: args
                .base_url
                .clone()
                .or_else(|| env_string("COGENT_BASE_URL")),
            version: env_string("COGENT_ANTHROPIC_VERSION")
                .or_else(|| env_string("ANTHROPIC_VERSION")),
            timeout,
        })?),
        ProviderPreset::Gemini
        | ProviderPreset::Deepseek
        | ProviderPreset::Qwen
        | ProviderPreset::Kimi
        | ProviderPreset::Glm
        | ProviderPreset::Proxy => {
            let base_url = args
                .base_url
                .clone()
                .or_else(|| env_string("COGENT_BASE_URL"))
                .or_else(|| provider.default_base_url().map(str::to_owned))
                .ok_or_else(|| "--base-url is required for --provider proxy".to_string())?;

            Ok(ProviderClient::proxy(ProxyConfig {
                base_url,
                auth: ProviderAuth::Bearer(SecretString::new(api_key)),
                protocol: ProxyProtocol::OpenAiCompatible,
                static_headers: parse_headers(&args.headers)?,
                timeout,
            })?)
        }
    }
}

fn resolve_provider(args: &Args) -> Result<ProviderPreset, Box<dyn std::error::Error>> {
    if let Some(provider) = args.provider {
        return Ok(provider);
    }
    if let Some(provider) = env_string("COGENT_PROVIDER") {
        return ProviderPreset::from_str(&provider, true)
            .map_err(|message| format!("invalid COGENT_PROVIDER={provider}: {message}").into());
    }
    Ok(ProviderPreset::Gemini)
}

fn resolve_api_key(
    args: &Args,
    provider: ProviderPreset,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(api_key) = &args.api_key {
        if !api_key.trim().is_empty() {
            return Ok(api_key.clone());
        }
    }
    if let Some(api_key) = env_string("COGENT_API_KEY") {
        return Ok(api_key);
    }

    let env_name = args
        .api_key_env
        .clone()
        .or_else(|| env_string("COGENT_API_KEY_ENV"))
        .unwrap_or_else(|| provider.default_api_key_env().to_string());
    let api_key = env::var(&env_name)
        .map_err(|_| format!("missing API key: set {env_name} in the environment or .env"))?;
    if api_key.trim().is_empty() {
        return Err(format!("{env_name} is set but empty").into());
    }
    Ok(api_key)
}

fn env_string(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_bool(name: &str) -> Result<Option<bool>, Box<dyn std::error::Error>> {
    let Some(value) = env_string(name) else {
        return Ok(None);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(format!("{name} must be true/false, got {value}").into()),
    }
}

fn env_u64(name: &str) -> Result<Option<u64>, Box<dyn std::error::Error>> {
    env_string(name)
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|err| format!("{name} must be an unsigned integer: {err}").into())
        })
        .transpose()
}

async fn stream_chat(
    client: &ProviderClient,
    request: ProviderChatRequest,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = client.stream_chat(request).await?;
    while let Some(event) = stream.next().await {
        match event? {
            ProviderStreamEvent::TokenBatch(batch) => {
                print!("{}", batch.text);
                io::stdout().flush()?;
            }
            ProviderStreamEvent::Usage { usage } => {
                eprintln!(
                    "\nusage input={:?} output={:?} total={:?}",
                    usage.input_tokens, usage.output_tokens, usage.total_tokens
                );
            }
            ProviderStreamEvent::Finished { finish_reason } => {
                eprintln!("\nfinish_reason={}", finish_reason.as_str());
            }
        }
    }
    Ok(())
}

fn parse_headers(values: &[String]) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    values
        .iter()
        .map(|value| {
            let (name, header_value) = value
                .split_once('=')
                .ok_or_else(|| format!("header must be formatted as name=value: {value}"))?;
            let name = name.trim();
            if name.is_empty() {
                return Err("header name must not be empty".to_string().into());
            }
            Ok((name.to_string(), header_value.trim().to_string()))
        })
        .collect()
}

fn load_env_file(path: &Path) -> io::Result<()> {
    if !path.exists() {
        return Ok(());
    }

    let contents = fs::read_to_string(path)?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || env::var_os(key).is_some() {
            continue;
        }

        env::set_var(key, unquote_env_value(value.trim()));
    }

    Ok(())
}

fn unquote_env_value(value: &str) -> String {
    let quoted = (value.starts_with('"') && value.ends_with('"'))
        || (value.starts_with('\'') && value.ends_with('\''));
    if quoted && value.len() >= 2 {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}
