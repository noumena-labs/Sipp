use std::{fmt, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Proxy,
    OpenAi,
    Anthropic,
}

impl ProviderKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Proxy => "proxy",
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn from_env(name: impl AsRef<str>) -> Result<Self, std::env::VarError> {
        std::env::var(name.as_ref()).map(Self)
    }

    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString([redacted])")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderAuth {
    Bearer(SecretString),
    Header { name: String, value: SecretString },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyProtocol {
    OpenAiCompatible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiConfig {
    pub api_key: SecretString,
    pub base_url: Option<String>,
    pub timeout: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicConfig {
    pub api_key: SecretString,
    pub base_url: Option<String>,
    pub version: Option<String>,
    pub timeout: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyConfig {
    pub base_url: String,
    pub auth: ProviderAuth,
    pub protocol: ProxyProtocol,
    pub static_headers: Vec<(String, String)>,
    pub timeout: Option<Duration>,
}
