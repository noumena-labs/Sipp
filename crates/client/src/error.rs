use thiserror::Error;

use crate::EndpointRef;

/// Result type used by the unified facade.
pub type CogentResult<T> = Result<T, CogentError>;

/// Remote service family that returned an error.
#[cfg(feature = "providers")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteKind {
    /// OpenAI-compatible proxy endpoint.
    Proxy,
    /// OpenAI API endpoint.
    OpenAi,
    /// Anthropic API endpoint.
    Anthropic,
}

#[cfg(feature = "providers")]
impl RemoteKind {
    /// Stable string used by bindings and diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Proxy => "proxy",
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

/// Classification for remote execution failures.
#[cfg(feature = "providers")]
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

#[cfg(feature = "providers")]
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
#[cfg(feature = "providers")]
#[derive(Debug, Clone, Error)]
#[error("{} remote error ({}): {message}", remote_kind.as_str(), kind.as_str())]
pub struct RemoteError {
    /// Error classification.
    pub kind: RemoteErrorKind,
    /// Remote service family that returned the error.
    pub remote_kind: RemoteKind,
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

#[cfg(feature = "providers")]
impl RemoteError {
    /// Create a remote error with no optional transport metadata.
    pub fn new(kind: RemoteErrorKind, remote_kind: RemoteKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            remote_kind,
            status: None,
            code: None,
            message: message.into(),
            retry_after: None,
            request_id: None,
            raw: None,
        }
    }
}

#[cfg(feature = "providers")]
impl From<cogentlm_providers::ProviderError> for RemoteError {
    fn from(error: cogentlm_providers::ProviderError) -> Self {
        Self {
            kind: match error.kind {
                cogentlm_providers::ProviderErrorKind::Authentication => {
                    RemoteErrorKind::Authentication
                }
                cogentlm_providers::ProviderErrorKind::Authorization => {
                    RemoteErrorKind::Authorization
                }
                cogentlm_providers::ProviderErrorKind::RateLimited => RemoteErrorKind::RateLimited,
                cogentlm_providers::ProviderErrorKind::QuotaExceeded => {
                    RemoteErrorKind::QuotaExceeded
                }
                cogentlm_providers::ProviderErrorKind::InvalidRequest => {
                    RemoteErrorKind::InvalidRequest
                }
                cogentlm_providers::ProviderErrorKind::UnsupportedFeature => {
                    RemoteErrorKind::UnsupportedFeature
                }
                cogentlm_providers::ProviderErrorKind::ModelNotFound => {
                    RemoteErrorKind::ModelNotFound
                }
                cogentlm_providers::ProviderErrorKind::Timeout => RemoteErrorKind::Timeout,
                cogentlm_providers::ProviderErrorKind::Overloaded => RemoteErrorKind::Overloaded,
                cogentlm_providers::ProviderErrorKind::Transport => RemoteErrorKind::Transport,
                cogentlm_providers::ProviderErrorKind::Provider => RemoteErrorKind::Remote,
            },
            remote_kind: match error.provider {
                cogentlm_providers::ProviderKind::Proxy => RemoteKind::Proxy,
                cogentlm_providers::ProviderKind::OpenAi => RemoteKind::OpenAi,
                cogentlm_providers::ProviderKind::Anthropic => RemoteKind::Anthropic,
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

#[cfg(feature = "providers")]
impl From<cogentlm_providers::ProviderError> for CogentError {
    fn from(error: cogentlm_providers::ProviderError) -> Self {
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
    #[cfg(feature = "providers")]
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
