use std::fmt;
use std::time::Duration;

use cogentlm_providers::{
    AnthropicAdapterConfig, OpenAiAdapterConfig, OpenAiCompatibleAdapterConfig,
    OpenAiCompatibleProtocol, ProviderAuth, ProviderTransport, SecretString,
};

use crate::{CogentError, CogentResult, ProviderEndpointError};

/// Redacted secret value used by direct provider configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct ProviderSecret(String);

impl ProviderSecret {
    /// Wrap a provider secret without exposing it through `Debug`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub(crate) fn expose(&self) -> &str {
        &self.0
    }

    fn as_gateway_secret(&self) -> SecretString {
        SecretString::new(self.expose().to_string())
    }
}

impl fmt::Debug for ProviderSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProviderSecret([redacted])")
    }
}

/// Authentication for an OpenAI-compatible direct provider endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderAuthConfig {
    /// Bearer token authentication.
    Bearer(ProviderSecret),
    /// Custom header authentication.
    Header {
        /// Header name.
        name: String,
        /// Header value.
        value: ProviderSecret,
    },
}

impl ProviderAuthConfig {
    fn secrets(&self) -> Vec<String> {
        match self {
            Self::Bearer(secret) => vec![secret.expose().to_string()],
            Self::Header { value, .. } => vec![value.expose().to_string()],
        }
    }

    fn into_provider_auth(self) -> ProviderAuth {
        match self {
            Self::Bearer(secret) => ProviderAuth::Bearer(secret.as_gateway_secret()),
            Self::Header { name, value } => ProviderAuth::Header {
                name,
                value: value.as_gateway_secret(),
            },
        }
    }
}

/// Direct provider endpoint descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderEndpointConfig {
    /// OpenAI API endpoint.
    OpenAi(OpenAiProviderConfig),
    /// Anthropic API endpoint.
    Anthropic(AnthropicProviderConfig),
    /// OpenAI-compatible API endpoint.
    OpenAiCompatible(OpenAiCompatibleProviderConfig),
}

impl ProviderEndpointConfig {
    /// Create an OpenAI provider config.
    pub fn openai(model: impl Into<String>, api_key: ProviderSecret) -> Self {
        Self::OpenAi(OpenAiProviderConfig {
            model: model.into(),
            api_key,
            base_url: None,
            timeout: None,
        })
    }

    /// Create an Anthropic provider config.
    pub fn anthropic(model: impl Into<String>, api_key: ProviderSecret) -> Self {
        Self::Anthropic(AnthropicProviderConfig {
            model: model.into(),
            api_key,
            base_url: None,
            version: None,
            timeout: None,
        })
    }

    /// Create an OpenAI-compatible provider config.
    pub fn openai_compatible(
        model: impl Into<String>,
        base_url: impl Into<String>,
        auth: ProviderAuthConfig,
    ) -> Self {
        Self::OpenAiCompatible(OpenAiCompatibleProviderConfig {
            model: model.into(),
            base_url: base_url.into(),
            auth,
            static_headers: Vec::new(),
            timeout: None,
        })
    }

    pub(crate) fn build(self) -> CogentResult<(String, ProviderTransport, Vec<String>)> {
        match self {
            Self::OpenAi(config) => build_openai_provider(config),
            Self::Anthropic(config) => build_anthropic_provider(config),
            Self::OpenAiCompatible(config) => build_openai_compatible_provider(config),
        }
    }
}

/// OpenAI direct provider configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiProviderConfig {
    /// Model name sent to OpenAI.
    pub model: String,
    /// OpenAI API key.
    pub api_key: ProviderSecret,
    /// Optional OpenAI-compatible base URL.
    pub base_url: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

/// Anthropic direct provider configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicProviderConfig {
    /// Model name sent to Anthropic.
    pub model: String,
    /// Anthropic API key.
    pub api_key: ProviderSecret,
    /// Optional Anthropic base URL.
    pub base_url: Option<String>,
    /// Optional Anthropic API version.
    pub version: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

/// OpenAI-compatible direct provider configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiCompatibleProviderConfig {
    /// Model name sent to the provider.
    pub model: String,
    /// Provider base URL.
    pub base_url: String,
    /// Provider authentication.
    pub auth: ProviderAuthConfig,
    /// Static provider headers. Values are treated as secrets.
    pub static_headers: Vec<(String, ProviderSecret)>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

fn build_openai_provider(
    config: OpenAiProviderConfig,
) -> CogentResult<(String, ProviderTransport, Vec<String>)> {
    let model = normalize_model(config.model)?;
    let secrets = vec![config.api_key.expose().to_string()];
    let transport = ProviderTransport::openai(OpenAiAdapterConfig {
        api_key: config.api_key.as_gateway_secret(),
        base_url: config.base_url,
        timeout: config.timeout,
    })
    .map_err(|error| {
        CogentError::Provider(ProviderEndpointError::from_provider_error(error, &secrets))
    })?;
    Ok((model, transport, secrets))
}

fn build_anthropic_provider(
    config: AnthropicProviderConfig,
) -> CogentResult<(String, ProviderTransport, Vec<String>)> {
    let model = normalize_model(config.model)?;
    let secrets = vec![config.api_key.expose().to_string()];
    let transport = ProviderTransport::anthropic(AnthropicAdapterConfig {
        api_key: config.api_key.as_gateway_secret(),
        base_url: config.base_url,
        version: config.version,
        timeout: config.timeout,
    })
    .map_err(|error| {
        CogentError::Provider(ProviderEndpointError::from_provider_error(error, &secrets))
    })?;
    Ok((model, transport, secrets))
}

fn build_openai_compatible_provider(
    config: OpenAiCompatibleProviderConfig,
) -> CogentResult<(String, ProviderTransport, Vec<String>)> {
    let model = normalize_model(config.model)?;
    let mut secrets = config.auth.secrets();
    secrets.extend(
        config
            .static_headers
            .iter()
            .map(|(_, value)| value.expose().to_string()),
    );
    let static_headers = config
        .static_headers
        .into_iter()
        .map(|(name, value)| (name, value.expose().to_string()))
        .collect();
    let transport = ProviderTransport::openai_compatible(OpenAiCompatibleAdapterConfig {
        base_url: config.base_url,
        auth: config.auth.into_provider_auth(),
        protocol: OpenAiCompatibleProtocol::OpenAiCompatible,
        static_headers,
        timeout: config.timeout,
    })
    .map_err(|error| {
        CogentError::Provider(ProviderEndpointError::from_provider_error(error, &secrets))
    })?;
    Ok((model, transport, secrets))
}

fn normalize_model(model: String) -> CogentResult<String> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        Err(CogentError::InvalidRequest(
            "provider model must not be empty".to_string(),
        ))
    } else if trimmed != model.as_str() {
        Err(CogentError::InvalidRequest(
            "provider model must not contain surrounding whitespace".to_string(),
        ))
    } else {
        Ok(model)
    }
}
