mod support;

use std::time::Duration;

use cogentlm::core::{ChatMessage, ChatRole};
use cogentlm::{
    CogentChatRequest, CogentClient, CogentTextOptions, EndpointDescriptor, ProviderAuthConfig,
    ProviderEndpointConfig, ProviderSecret,
};
use futures::executor::block_on;

const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/openai/";
const GEMINI_DEFAULT_MODEL: &str = "gemini-3.5-flash";
const OPENAI_DEFAULT_MODEL: &str = "gpt-5-mini";

fn main() -> support::ExampleResult<()> {
    block_on(async {
        let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
        let prompt = if prompt.is_empty() {
            "Say hello from a direct provider.".to_string()
        } else {
            prompt
        };

        // Direct providers belong in trusted Rust processes. Browser code
        // should call a gateway or application route instead of holding
        // provider credentials.
        let mut client = CogentClient::new();
        let endpoint = client
            .add("provider", EndpointDescriptor::provider(provider_config()?))
            .await?;
        let response = client
            .chat(CogentChatRequest {
                endpoint: Some(endpoint),
                messages: vec![ChatMessage::new(ChatRole::User, prompt)],
                options: text_options(),
                ..Default::default()
            })
            .await?;
        support::print_text(response);
        Ok(())
    })
}

fn provider_config() -> support::ExampleResult<ProviderEndpointConfig> {
    let config = match provider_name().as_str() {
        "gemini" => ProviderEndpointConfig::openai_compatible(
            env_any(
                &["COGENTLM_PROVIDER_MODEL", "GEMINI_MODEL"],
                Some(GEMINI_DEFAULT_MODEL),
            )?,
            support::optional_env("COGENTLM_PROVIDER_BASE_URL")
                .unwrap_or_else(|| GEMINI_BASE_URL.to_string()),
            ProviderAuthConfig::Bearer(ProviderSecret::new(required_env_any(&[
                "COGENTLM_PROVIDER_API_KEY",
                "GEMINI_API_KEY",
            ])?)),
        ),
        "openai" => ProviderEndpointConfig::openai(
            env_any(
                &["COGENTLM_PROVIDER_MODEL", "OPENAI_MODEL"],
                Some(OPENAI_DEFAULT_MODEL),
            )?,
            ProviderSecret::new(required_env_any(&[
                "COGENTLM_PROVIDER_API_KEY",
                "OPENAI_API_KEY",
            ])?),
        ),
        "anthropic" => ProviderEndpointConfig::anthropic(
            required_env_any(&["COGENTLM_PROVIDER_MODEL", "ANTHROPIC_MODEL"])?,
            ProviderSecret::new(required_env_any(&[
                "COGENTLM_PROVIDER_API_KEY",
                "ANTHROPIC_API_KEY",
            ])?),
        ),
        "openai_compatible" => ProviderEndpointConfig::openai_compatible(
            required_env_any(&["COGENTLM_PROVIDER_MODEL"])?,
            required_env_any(&["COGENTLM_PROVIDER_BASE_URL"])?,
            openai_compatible_auth()?,
        ),
        _ => {
            return Err(config_error(
                "COGENTLM_PROVIDER must be gemini, openai, anthropic, or openai_compatible",
            ));
        }
    };
    Ok(with_optional_provider_config(config))
}

fn with_optional_provider_config(mut config: ProviderEndpointConfig) -> ProviderEndpointConfig {
    let timeout = Some(Duration::from_millis(
        support::env_parse("COGENTLM_PROVIDER_TIMEOUT_MS").unwrap_or(30_000),
    ));
    match &mut config {
        ProviderEndpointConfig::OpenAi(config) => {
            config.base_url = support::optional_env("COGENTLM_PROVIDER_BASE_URL")
                .or_else(|| support::optional_env("OPENAI_BASE_URL"));
            config.timeout = timeout;
        }
        ProviderEndpointConfig::Anthropic(config) => {
            config.base_url = support::optional_env("COGENTLM_PROVIDER_BASE_URL")
                .or_else(|| support::optional_env("ANTHROPIC_BASE_URL"));
            config.version = support::optional_env("ANTHROPIC_VERSION");
            config.timeout = timeout;
        }
        ProviderEndpointConfig::OpenAiCompatible(config) => {
            config.timeout = timeout;
        }
    }
    config
}

fn openai_compatible_auth() -> support::ExampleResult<ProviderAuthConfig> {
    match (
        support::optional_env("COGENTLM_PROVIDER_AUTH_HEADER_NAME"),
        support::optional_env("COGENTLM_PROVIDER_AUTH_HEADER_VALUE"),
    ) {
        (Some(name), Some(value)) => Ok(ProviderAuthConfig::Header {
            name,
            value: ProviderSecret::new(value),
        }),
        (None, None) => Ok(ProviderAuthConfig::Bearer(ProviderSecret::new(
            required_env_any(&["COGENTLM_PROVIDER_API_KEY"])?,
        ))),
        _ => Err(config_error(
            "COGENTLM_PROVIDER_AUTH_HEADER_NAME and \
             COGENTLM_PROVIDER_AUTH_HEADER_VALUE must be set together",
        )),
    }
}

fn text_options() -> CogentTextOptions {
    CogentTextOptions {
        max_tokens: support::env_parse("COGENTLM_MAX_TOKENS").or(Some(support::DEFAULT_MAX_TOKENS)),
        temperature: support::env_parse("COGENTLM_TEMPERATURE")
            .or(Some(support::DEFAULT_TEMPERATURE)),
        top_p: support::env_parse("COGENTLM_TOP_P").or(Some(support::DEFAULT_TOP_P)),
        stop: Vec::new(),
    }
}

fn provider_name() -> String {
    support::optional_env("COGENTLM_PROVIDER")
        .unwrap_or_else(|| "gemini".to_string())
        .to_lowercase()
        .replace('-', "_")
}

fn env_any(
    names: &[&'static str],
    default: Option<&'static str>,
) -> support::ExampleResult<String> {
    for name in names {
        if let Some(value) = support::optional_env(name) {
            return Ok(value);
        }
    }
    default
        .map(str::to_string)
        .ok_or_else(|| config_error(format!("{} is required", names.join(" or "))))
}

fn required_env_any(names: &[&'static str]) -> support::ExampleResult<String> {
    env_any(names, None)
}

fn config_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}
