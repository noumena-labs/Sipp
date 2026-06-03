use thiserror::Error;

#[cfg(feature = "remote")]
use cogentlm_remote::{GatewayError, GatewayErrorKind};

use crate::EndpointRef;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/error_tests.rs"]
mod error_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Result type used by the unified facade.
pub type CogentResult<T> = Result<T, CogentError>;

/// Classification for remote execution failures.
#[cfg(feature = "remote")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteErrorKind {
    /// Authentication failed.
    Authentication,
    /// Authorization failed after authentication.
    Authorization,
    /// The remote service rate limited the request.
    RateLimited,
    /// The remote account quota is exhausted.
    QuotaExceeded,
    /// The request is invalid for the remote service.
    InvalidRequest,
    /// The requested feature is not supported by the remote service.
    UnsupportedFeature,
    /// The requested model was not found by the remote service.
    ModelNotFound,
    /// The remote request timed out.
    Timeout,
    /// The remote service is overloaded.
    Overloaded,
    /// Network or protocol transport failed.
    Transport,
    /// Remote service returned an unclassified API error.
    Remote,
}

#[cfg(feature = "remote")]
impl RemoteErrorKind {
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
            Self::Remote => "remote",
        }
    }
}

/// Structured error returned by remote endpoints.
#[cfg(feature = "remote")]
#[derive(Debug, Clone, Error)]
#[error("remote gateway error ({}): {message}", kind.as_str())]
pub struct RemoteError {
    /// Error classification.
    pub kind: RemoteErrorKind,
    /// HTTP status code when the remote returned one.
    pub status: Option<u16>,
    /// Remote service-specific error code when available.
    pub code: Option<String>,
    /// Human-readable error message.
    pub message: String,
    /// Retry delay returned by the remote service.
    pub retry_after: Option<std::time::Duration>,
    /// Remote request id when available.
    pub request_id: Option<String>,
    /// Raw remote error payload when available.
    pub raw: Option<Box<serde_json::Value>>,
}

#[cfg(feature = "remote")]
impl RemoteError {
    /// Create a remote error with no optional transport metadata.
    pub fn new(kind: RemoteErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            status: None,
            code: None,
            message: message.into(),
            retry_after: None,
            request_id: None,
            raw: None,
        }
    }
}

#[cfg(feature = "remote")]
impl From<GatewayError> for RemoteError {
    fn from(error: GatewayError) -> Self {
        Self {
            kind: match error.kind {
                GatewayErrorKind::Authentication => RemoteErrorKind::Authentication,
                GatewayErrorKind::Authorization => RemoteErrorKind::Authorization,
                GatewayErrorKind::RateLimited => RemoteErrorKind::RateLimited,
                GatewayErrorKind::QuotaExceeded => RemoteErrorKind::QuotaExceeded,
                GatewayErrorKind::InvalidRequest => RemoteErrorKind::InvalidRequest,
                GatewayErrorKind::UnsupportedFeature => RemoteErrorKind::UnsupportedFeature,
                GatewayErrorKind::ModelNotFound => RemoteErrorKind::ModelNotFound,
                GatewayErrorKind::Timeout => RemoteErrorKind::Timeout,
                GatewayErrorKind::Overloaded => RemoteErrorKind::Overloaded,
                GatewayErrorKind::Transport => RemoteErrorKind::Transport,
                GatewayErrorKind::Gateway => RemoteErrorKind::Remote,
            },
            status: error.status,
            code: error.code,
            message: error.message,
            retry_after: error.retry_after,
            request_id: error.request_id,
            raw: error.raw,
        }
    }
}

#[cfg(feature = "remote")]
impl From<GatewayError> for CogentError {
    fn from(error: GatewayError) -> Self {
        Self::Remote(RemoteError::from(error))
    }
}

/// Error type for endpoint resolution, validation, and endpoint execution.
#[derive(Debug, Error)]
pub enum CogentError {
    /// Local engine error.
    #[error(transparent)]
    Local(#[from] cogentlm_engine::Error),

    /// Remote endpoint error.
    #[cfg(feature = "remote")]
    #[error(transparent)]
    Remote(RemoteError),

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
