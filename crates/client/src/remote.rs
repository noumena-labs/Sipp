use std::fmt;
use std::time::Duration;

use cogentlm_providers::{
    AnthropicConfig, OpenAiConfig, ProviderAuth, ProviderTransport, ProxyConfig, ProxyProtocol,
    SecretString,
};

use crate::CogentResult;

/// Redacted secret value used by remote endpoint configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct RemoteSecret(String);

impl RemoteSecret {
    /// Wrap a secret without exposing it through `Debug`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for RemoteSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RemoteSecret([redacted])")
    }
}

/// Authentication used by OpenAI-compatible proxy remotes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteAuth {
    /// Bearer token sent as an authorization header.
    Bearer(RemoteSecret),
    /// Custom header name and secret value.
    Header {
        /// Header name.
        name: String,
        /// Header value.
        value: RemoteSecret,
    },
}

impl RemoteAuth {
    fn to_provider(&self) -> ProviderAuth {
        match self {
            Self::Bearer(secret) => {
                ProviderAuth::Bearer(SecretString::new(secret.expose().to_string()))
            }
            Self::Header { name, value } => ProviderAuth::Header {
                name: name.clone(),
                value: SecretString::new(value.expose().to_string()),
            },
        }
    }
}

/// Wire protocol used by a generic proxy remote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteProtocol {
    /// OpenAI-compatible chat, completion, and embedding endpoints.
    OpenAiCompatible,
}

impl From<RemoteProtocol> for ProxyProtocol {
    fn from(value: RemoteProtocol) -> Self {
        match value {
            RemoteProtocol::OpenAiCompatible => Self::OpenAiCompatible,
        }
    }
}

/// Configuration for a remote inference endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteConfig {
    /// Native OpenAI API endpoint.
    OpenAi(RemoteOpenAiConfig),
    /// Native Anthropic API endpoint.
    Anthropic(RemoteAnthropicConfig),
    /// Generic OpenAI-compatible proxy endpoint.
    Proxy(RemoteProxyConfig),
}

impl RemoteConfig {
    /// Create an OpenAI remote with default base URL and timeout.
    pub fn openai(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::OpenAi(RemoteOpenAiConfig {
            model: model.into(),
            api_key: RemoteSecret::new(api_key),
            base_url: None,
            timeout: None,
        })
    }

    /// Create an Anthropic remote with default base URL and timeout.
    pub fn anthropic(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::Anthropic(RemoteAnthropicConfig {
            model: model.into(),
            api_key: RemoteSecret::new(api_key),
            base_url: None,
            version: None,
            timeout: None,
        })
    }

    /// Create an OpenAI-compatible proxy remote.
    pub fn proxy(model: impl Into<String>, base_url: impl Into<String>, auth: RemoteAuth) -> Self {
        Self::Proxy(RemoteProxyConfig {
            model: model.into(),
            base_url: base_url.into(),
            auth,
            protocol: RemoteProtocol::OpenAiCompatible,
            static_headers: Vec::new(),
            timeout: None,
        })
    }

    pub(crate) fn build(self) -> CogentResult<(String, ProviderTransport)> {
        match self {
            Self::OpenAi(config) => {
                let model = config.model;
                let transport = ProviderTransport::openai(OpenAiConfig {
                    api_key: SecretString::new(config.api_key.expose().to_string()),
                    base_url: config.base_url,
                    timeout: config.timeout,
                })?;
                Ok((model, transport))
            }
            Self::Anthropic(config) => {
                let model = config.model;
                let transport = ProviderTransport::anthropic(AnthropicConfig {
                    api_key: SecretString::new(config.api_key.expose().to_string()),
                    base_url: config.base_url,
                    version: config.version,
                    timeout: config.timeout,
                })?;
                Ok((model, transport))
            }
            Self::Proxy(config) => {
                let model = config.model;
                let transport = ProviderTransport::proxy(ProxyConfig {
                    base_url: config.base_url,
                    auth: config.auth.to_provider(),
                    protocol: config.protocol.into(),
                    static_headers: config.static_headers,
                    timeout: config.timeout,
                })?;
                Ok((model, transport))
            }
        }
    }
}

/// OpenAI remote endpoint configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteOpenAiConfig {
    /// Remote model id.
    pub model: String,
    /// OpenAI API key.
    pub api_key: RemoteSecret,
    /// Optional override for the OpenAI API base URL.
    pub base_url: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

/// Anthropic remote endpoint configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteAnthropicConfig {
    /// Remote model id.
    pub model: String,
    /// Anthropic API key.
    pub api_key: RemoteSecret,
    /// Optional override for the Anthropic API base URL.
    pub base_url: Option<String>,
    /// Optional Anthropic API version override.
    pub version: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

/// Generic proxy remote endpoint configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteProxyConfig {
    /// Remote model id.
    pub model: String,
    /// Proxy base URL.
    pub base_url: String,
    /// Proxy authentication.
    pub auth: RemoteAuth,
    /// Proxy wire protocol.
    pub protocol: RemoteProtocol,
    /// Static headers added to every proxy request.
    pub static_headers: Vec<(String, String)>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}
