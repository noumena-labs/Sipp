use std::{fmt, time::Duration};

/// Server-side adapter kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// OpenAI-compatible adapter.
    OpenAiCompatible,
    /// OpenAI adapter.
    OpenAi,
    /// Anthropic adapter.
    Anthropic,
}

impl ProviderKind {
    /// Stable adapter label used in metadata and diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "openai_compatible",
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

/// Redacted secret string.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    /// Wrap a secret value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Load a secret from an environment variable.
    pub fn from_env(name: impl AsRef<str>) -> Result<Self, std::env::VarError> {
        std::env::var(name.as_ref()).map(Self)
    }

    /// Expose the raw secret to adapter internals.
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub(crate) fn is_blank(&self) -> bool {
        self.0.trim().is_empty()
    }

    pub(crate) fn contains_whitespace(&self) -> bool {
        self.0.chars().any(char::is_whitespace)
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString([redacted])")
    }
}

/// Provider authentication owned by provider adapter configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderAuth {
    /// Bearer token authentication.
    Bearer(SecretString),
    /// Custom header authentication.
    Header { name: String, value: SecretString },
}

/// Wire protocol spoken by an OpenAI-compatible adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiCompatibleProtocol {
    /// OpenAI-compatible chat, embeddings, models, and streaming routes.
    OpenAiCompatible,
}

/// OpenAI adapter configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiAdapterConfig {
    /// OpenAI API key.
    pub api_key: SecretString,
    /// Optional OpenAI-compatible base URL.
    pub base_url: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

/// Anthropic adapter configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicAdapterConfig {
    /// Anthropic API key.
    pub api_key: SecretString,
    /// Optional Anthropic base URL.
    pub base_url: Option<String>,
    /// Optional Anthropic API version.
    pub version: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

/// OpenAI-compatible adapter configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct OpenAiCompatibleAdapterConfig {
    /// Provider-compatible base URL.
    pub base_url: String,
    /// Adapter authentication.
    pub auth: ProviderAuth,
    /// Wire protocol.
    pub protocol: OpenAiCompatibleProtocol,
    /// Static headers owned by provider configuration.
    pub static_headers: Vec<(String, String)>,
    /// Optional per-request correlation header.
    pub correlation_header: Option<String>,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

impl fmt::Debug for OpenAiCompatibleAdapterConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let static_headers = self
            .static_headers
            .iter()
            .map(|(name, _)| (name, "[redacted]"))
            .collect::<Vec<_>>();
        f.debug_struct("OpenAiCompatibleAdapterConfig")
            .field("base_url", &self.base_url)
            .field("auth", &self.auth)
            .field("protocol", &self.protocol)
            .field("static_headers", &static_headers)
            .field("correlation_header", &self.correlation_header)
            .field("timeout", &self.timeout)
            .finish()
    }
}
