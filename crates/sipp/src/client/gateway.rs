use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

/// Secret configuration value for gateway endpoints.
#[derive(Clone, PartialEq, Eq)]
pub struct GatewaySecret(String);

impl GatewaySecret {
    /// Wrap a secret value without exposing it through `Debug`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for GatewaySecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GatewaySecret([redacted])")
    }
}

/// Authentication applied to outbound gateway endpoint requests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum GatewayAuthentication {
    /// Send no authentication header.
    #[default]
    None,
    /// Send an HTTP bearer token.
    Bearer(GatewaySecret),
    /// Send a caller-defined static header.
    Header {
        /// Header name.
        name: String,
        /// Sensitive header value.
        value: GatewaySecret,
    },
}

/// Timeouts applied by a gateway endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayTimeoutPolicy {
    /// Connection establishment deadline.
    pub connect: Duration,
    /// Total finite-request deadline.
    pub request: Duration,
    /// Streaming idle-read deadline.
    pub read: Duration,
}

impl Default for GatewayTimeoutPolicy {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            request: Duration::from_secs(60),
            read: Duration::from_secs(60),
        }
    }
}

/// Configurable first-party gateway routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayRoutes {
    /// Query inference route.
    pub query: String,
    /// Chat inference route.
    pub chat: String,
    /// Embedding inference route.
    pub embed: String,
    /// Optional model or target index route.
    pub index: Option<String>,
    /// Optional liveness route.
    pub health: Option<String>,
    /// Optional readiness route.
    pub readiness: Option<String>,
    /// Optional metrics route.
    pub metrics: Option<String>,
}

impl Default for GatewayRoutes {
    fn default() -> Self {
        Self {
            query: "/v1/query".to_string(),
            chat: "/v1/chat".to_string(),
            embed: "/v1/embed".to_string(),
            index: Some("/v1/models".to_string()),
            health: Some("/health".to_string()),
            readiness: Some("/ready".to_string()),
            metrics: Some("/metrics".to_string()),
        }
    }
}

/// Construction parameters for a client-owned gateway endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayEndpointConfig {
    /// Target name encoded in gateway profile requests.
    pub target: String,
    /// Absolute HTTP(S) service URL.
    pub base_url: String,
    /// Route selection.
    pub routes: GatewayRoutes,
    /// Authentication strategy.
    pub authentication: GatewayAuthentication,
    /// Additional static request headers.
    pub static_headers: BTreeMap<String, String>,
    /// HTTP timeout policy.
    pub timeouts: GatewayTimeoutPolicy,
    /// Profile-specific options merged into request bodies.
    pub protocol_options: serde_json::Map<String, serde_json::Value>,
}
