use std::fmt;
use std::time::Duration;

use cogentlm_remote::{GatewayConfig, GatewaySecret, GatewayTransport};

use crate::{CogentError, CogentResult};

/// Redacted secret value used by remote gateway configuration.
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

/// Configuration for a CogentLM remote gateway endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteGatewayConfig {
    /// Public gateway alias.
    pub alias: String,
    /// Gateway base URL.
    pub base_url: String,
    /// Bearer token issued for this gateway.
    pub token: RemoteSecret,
    /// Request timeout.
    pub timeout: Option<Duration>,
}

impl RemoteGatewayConfig {
    pub(crate) fn build(self) -> CogentResult<(String, GatewayTransport)> {
        let alias = normalize_alias(self.alias)?;
        let transport = GatewayTransport::new(GatewayConfig {
            base_url: self.base_url,
            token: GatewaySecret::new(self.token.expose().to_string()),
            timeout: self.timeout,
        })?;
        Ok((alias, transport))
    }
}

fn normalize_alias(alias: String) -> CogentResult<String> {
    let trimmed = alias.trim();
    if trimmed.is_empty() {
        Err(CogentError::InvalidRequest(
            "remote alias must not be empty".to_string(),
        ))
    } else if trimmed != alias.as_str() {
        Err(CogentError::InvalidRequest(
            "remote alias must not contain surrounding whitespace".to_string(),
        ))
    } else {
        Ok(alias)
    }
}
