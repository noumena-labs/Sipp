use std::{fmt, time::Duration};

/// Redacted bearer token used by gateway transport configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct GatewaySecret(String);

impl GatewaySecret {
    /// Wrap a secret without exposing it through `Debug`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub(crate) fn expose(&self) -> &str {
        &self.0
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for GatewaySecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GatewaySecret([redacted])")
    }
}

/// Client-side configuration for a CogentLM remote gateway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayConfig {
    /// Gateway base URL.
    pub base_url: String,
    /// Bearer token issued for this gateway.
    pub token: GatewaySecret,
    /// Request timeout used for connection setup, idle reads, and unary calls.
    pub timeout: Option<Duration>,
}
