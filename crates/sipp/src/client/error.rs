use thiserror::Error;

use crate::client::SippCancellationReason;
use crate::client::EndpointRef;
#[cfg(feature = "providers")]
use crate::providers::{ProviderError, ProviderErrorKind};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/client/error_tests.rs"]
mod error_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Result type used by the unified facade.
pub type SippResult<T> = Result<T, SippError>;

/// Structured error returned by a gateway endpoint.
#[derive(Debug, Clone, Error)]
#[error("endpoint error ({kind}): {message}")]
pub struct EndpointError {
    /// Endpoint-defined stable error classification.
    pub kind: String,
    /// Transport status code when the endpoint returned one.
    pub status: Option<u16>,
    /// Endpoint-specific error code when available.
    pub code: Option<String>,
    /// Human-readable error message.
    pub message: String,
    /// Retry delay returned by the endpoint.
    pub retry_after: Option<std::time::Duration>,
    /// Upstream request id when available.
    pub request_id: Option<String>,
    /// Raw endpoint error payload when available.
    pub raw: Option<Box<serde_json::Value>>,
}

impl EndpointError {
    /// Create an endpoint error with no optional transport metadata.
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            status: None,
            code: None,
            message: message.into(),
            retry_after: None,
            request_id: None,
            raw: None,
        }
    }
}

/// Classification for direct provider execution failures.
#[cfg(feature = "providers")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderEndpointErrorKind {
    /// Authentication failed.
    Authentication,
    /// Authorization failed after authentication.
    Authorization,
    /// The provider rate limited the request.
    RateLimited,
    /// The provider account quota is exhausted.
    QuotaExceeded,
    /// The request is invalid for the provider.
    InvalidRequest,
    /// The requested feature is not supported by the provider.
    UnsupportedFeature,
    /// The requested model was not found by the provider.
    ModelNotFound,
    /// The provider request timed out.
    Timeout,
    /// The provider service is overloaded.
    Overloaded,
    /// Network or protocol transport failed.
    Transport,
    /// Provider returned an unclassified API error.
    Provider,
}

#[cfg(feature = "providers")]
impl ProviderEndpointErrorKind {
    /// Stable string used by bindings and diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::Authorization => "authorization",
            Self::RateLimited => "rate_limited",
            Self::QuotaExceeded => "quota_exceeded",
            Self::InvalidRequest => "invalid_request",
            Self::UnsupportedFeature => "unsupported_feature",
            Self::ModelNotFound => "model_not_found",
            Self::Timeout => "timeout",
            Self::Overloaded => "overloaded",
            Self::Transport => "transport",
            Self::Provider => "provider",
        }
    }
}

/// Structured error returned by direct provider endpoints.
#[cfg(feature = "providers")]
#[derive(Debug, Clone, Error)]
#[error("provider error ({} {provider}): {message}", kind.as_str())]
pub struct ProviderEndpointError {
    /// Error classification.
    pub kind: ProviderEndpointErrorKind,
    /// Provider label.
    pub provider: String,
    /// HTTP status code when the provider returned one.
    pub status: Option<u16>,
    /// Provider-specific error code when available.
    pub code: Option<String>,
    /// Human-readable error message with configured secrets redacted.
    pub message: String,
    /// Retry delay returned by the provider.
    pub retry_after: Option<std::time::Duration>,
    /// Provider request id when available.
    pub request_id: Option<String>,
    /// Raw provider error payload with configured secrets redacted.
    pub raw: Option<Box<serde_json::Value>>,
}

#[cfg(feature = "providers")]
impl ProviderEndpointError {
    /// Create a provider error with no optional transport metadata.
    pub fn new(
        kind: ProviderEndpointErrorKind,
        provider: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            provider: provider.into(),
            status: None,
            code: None,
            message: message.into(),
            retry_after: None,
            request_id: None,
            raw: None,
        }
    }

    pub(crate) fn from_provider_error(error: ProviderError, secrets: &[String]) -> Self {
        Self {
            kind: match error.kind {
                ProviderErrorKind::Authentication => ProviderEndpointErrorKind::Authentication,
                ProviderErrorKind::Authorization => ProviderEndpointErrorKind::Authorization,
                ProviderErrorKind::RateLimited => ProviderEndpointErrorKind::RateLimited,
                ProviderErrorKind::QuotaExceeded => ProviderEndpointErrorKind::QuotaExceeded,
                ProviderErrorKind::InvalidRequest => ProviderEndpointErrorKind::InvalidRequest,
                ProviderErrorKind::UnsupportedFeature => {
                    ProviderEndpointErrorKind::UnsupportedFeature
                }
                ProviderErrorKind::ModelNotFound => ProviderEndpointErrorKind::ModelNotFound,
                ProviderErrorKind::Timeout => ProviderEndpointErrorKind::Timeout,
                ProviderErrorKind::Overloaded => ProviderEndpointErrorKind::Overloaded,
                ProviderErrorKind::Transport => ProviderEndpointErrorKind::Transport,
                ProviderErrorKind::Provider => ProviderEndpointErrorKind::Provider,
            },
            provider: error.provider.as_str().to_string(),
            status: error.status,
            code: error.code.map(|value| redact_string(value, secrets)),
            message: redact_string(error.message, secrets),
            retry_after: error.retry_after,
            request_id: error.request_id.map(|value| redact_string(value, secrets)),
            raw: error
                .raw
                .map(|value| Box::new(redact_json_value(*value, secrets))),
        }
    }
}

/// Error type for endpoint resolution, validation, and endpoint execution.
#[derive(Debug, Error)]
pub enum SippError {
    /// Local engine error.
    #[error(transparent)]
    Local(#[from] crate::error::Error),

    /// Gateway endpoint error.
    #[error(transparent)]
    Endpoint(EndpointError),

    /// Direct provider endpoint error.
    #[cfg(feature = "providers")]
    #[error(transparent)]
    Provider(ProviderEndpointError),

    /// The caller cancelled an in-flight request.
    #[error("request cancelled ({})", reason.as_str())]
    Cancelled {
        /// Stable cancellation classification.
        reason: SippCancellationReason,
    },

    /// Internal client error.
    #[error("internal facade error: {0}")]
    Internal(String),

    /// Requested endpoint id has not been registered.
    #[error("endpoint not found: {0:?}")]
    EndpointNotFound(EndpointRef),

    /// No endpoint was specified and multiple endpoints support the operation.
    #[error("ambiguous endpoint for {operation}")]
    AmbiguousEndpoint {
        /// Requested operation.
        operation: &'static str,
    },

    /// No endpoint supports the requested operation.
    #[error("no supported endpoint for {operation}")]
    NoSupportedEndpoint {
        /// Requested operation.
        operation: &'static str,
    },

    /// The selected endpoint cannot run the requested operation.
    #[error("unsupported operation {operation} on endpoint {endpoint:?}")]
    UnsupportedOperation {
        /// Selected endpoint.
        endpoint: EndpointRef,
        /// Requested operation.
        operation: &'static str,
    },

    /// Request validation failed before execution.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

#[cfg(feature = "providers")]
fn redact_string(mut value: String, secrets: &[String]) -> String {
    for secret in secrets {
        if secret.is_empty() {
            continue;
        }
        value = value.replace(secret, "[redacted]");
    }
    value
}

#[cfg(feature = "providers")]
fn redact_json_value(value: serde_json::Value, secrets: &[String]) -> serde_json::Value {
    match value {
        serde_json::Value::String(value) => {
            serde_json::Value::String(redact_string(value, secrets))
        }
        serde_json::Value::Array(values) => serde_json::Value::Array(
            values
                .into_iter()
                .map(|value| redact_json_value(value, secrets))
                .collect(),
        ),
        serde_json::Value::Object(values) => serde_json::Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, redact_json_value(value, secrets)))
                .collect(),
        ),
        value => value,
    }
}
