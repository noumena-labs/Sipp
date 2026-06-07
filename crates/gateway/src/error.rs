use std::time::Duration;

use thiserror::Error;

/// Result type returned by gateway server operations.
pub type GatewayResult<T> = Result<T, GatewayError>;

/// Gateway error classification exposed through normalized HTTP errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayErrorKind {
    /// Missing or invalid gateway bearer token.
    Authentication,
    /// Authenticated caller is not allowed to use this alias or operation.
    Authorization,
    /// Caller has exceeded rate limits.
    RateLimited,
    /// Caller has exhausted quota.
    QuotaExceeded,
    /// Request body or gateway configuration is invalid.
    InvalidRequest,
    /// Request body exceeded the gateway size limit.
    RequestTooLarge,
    /// Alias does not support the requested operation.
    UnsupportedFeature,
    /// Public alias was not found.
    ModelNotFound,
    /// Gateway or upstream operation timed out.
    Timeout,
    /// Gateway or upstream backend is overloaded.
    Overloaded,
    /// The server is restarting or draining.
    ServerRestarting,
    /// The downstream client disconnected.
    ClientDisconnected,
    /// The caller explicitly cancelled the request.
    CallerCancelled,
    /// A request deadline expired.
    DeadlineExceeded,
    /// Network transport to a backend failed.
    Transport,
    /// Gateway internal failure.
    Internal,
}

impl GatewayErrorKind {
    /// Stable gateway error code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::Authorization => "authorization",
            Self::RateLimited => "rate_limited",
            Self::QuotaExceeded => "quota_exceeded",
            Self::InvalidRequest => "invalid_request",
            Self::RequestTooLarge => "request_too_large",
            Self::UnsupportedFeature => "unsupported_feature",
            Self::ModelNotFound => "model_not_found",
            Self::Timeout => "timeout",
            Self::Overloaded => "overloaded",
            Self::ServerRestarting => "server_restarting",
            Self::ClientDisconnected => "client_disconnected",
            Self::CallerCancelled => "caller_cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::Transport => "transport",
            Self::Internal => "internal",
        }
    }

    /// HTTP status code commonly used by CogentLM gateway servers.
    pub const fn http_status_code(self) -> u16 {
        match self {
            Self::Authentication => 401,
            Self::Authorization => 403,
            Self::RateLimited => 429,
            Self::QuotaExceeded => 402,
            Self::InvalidRequest | Self::UnsupportedFeature => 400,
            Self::RequestTooLarge => 413,
            Self::ModelNotFound => 404,
            Self::Timeout | Self::DeadlineExceeded => 408,
            Self::Overloaded | Self::ServerRestarting => 503,
            Self::ClientDisconnected | Self::CallerCancelled => 499,
            Self::Transport | Self::Internal => 500,
        }
    }
}

/// Normalized gateway error.
#[derive(Debug, Clone, Error)]
#[error("gateway error ({}): {message}", kind.as_str())]
pub struct GatewayError {
    /// Error classification.
    pub kind: GatewayErrorKind,
    /// Human-readable message safe to return to clients.
    pub message: String,
    /// Retry delay when applicable.
    pub retry_after: Option<Duration>,
    /// Upstream provider or gateway request ID, when available.
    pub upstream_request_id: Option<String>,
}

impl GatewayError {
    /// Create a normalized gateway error.
    pub fn new(kind: GatewayErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retry_after: None,
            upstream_request_id: None,
        }
    }

    /// Stable gateway error code derived from the classification.
    pub const fn code(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Attach retry-after metadata to the error.
    pub fn with_retry_after(mut self, retry_after: Option<Duration>) -> Self {
        self.retry_after = retry_after;
        self
    }

    /// Attach an upstream request identifier for structured logging.
    pub fn with_upstream_request_id(mut self, request_id: Option<String>) -> Self {
        self.upstream_request_id = request_id;
        self
    }
}

impl From<cogentlm_client::CogentError> for GatewayError {
    fn from(error: cogentlm_client::CogentError) -> Self {
        use cogentlm_client::{CogentCancellationReason, CogentError};

        match error {
            CogentError::Cancelled { reason } => match reason {
                CogentCancellationReason::ClientDisconnected => {
                    Self::new(GatewayErrorKind::ClientDisconnected, "client disconnected")
                }
                CogentCancellationReason::ServerShutdown => {
                    Self::new(GatewayErrorKind::ServerRestarting, "server is restarting")
                }
                CogentCancellationReason::CallerCancelled => {
                    Self::new(GatewayErrorKind::CallerCancelled, "request cancelled")
                }
                CogentCancellationReason::DeadlineExceeded => Self::new(
                    GatewayErrorKind::DeadlineExceeded,
                    "request deadline exceeded",
                ),
            },
            #[cfg(feature = "remote")]
            CogentError::Remote(error) => Self::new(
                remote_error_kind(error.kind),
                "upstream gateway request failed",
            )
            .with_retry_after(error.retry_after)
            .with_upstream_request_id(error.request_id),
            #[cfg(feature = "providers")]
            CogentError::Provider(error) => Self::new(
                provider_error_kind(error.kind),
                "upstream provider request failed",
            )
            .with_retry_after(error.retry_after)
            .with_upstream_request_id(error.request_id),
            CogentError::EndpointNotFound(_) => {
                Self::new(GatewayErrorKind::ModelNotFound, "model alias not found")
            }
            CogentError::UnsupportedOperation { .. }
            | CogentError::AmbiguousEndpoint { .. }
            | CogentError::NoSupportedEndpoint { .. }
            | CogentError::InvalidRequest(_) => {
                Self::new(GatewayErrorKind::InvalidRequest, error.to_string())
            }
            CogentError::Local(_) | CogentError::Internal(_) => {
                Self::new(GatewayErrorKind::Internal, "gateway execution failed")
            }
        }
    }
}

#[cfg(feature = "remote")]
fn remote_error_kind(kind: cogentlm_client::RemoteErrorKind) -> GatewayErrorKind {
    use cogentlm_client::RemoteErrorKind;
    match kind {
        RemoteErrorKind::Authentication => GatewayErrorKind::Authentication,
        RemoteErrorKind::Authorization => GatewayErrorKind::Authorization,
        RemoteErrorKind::RateLimited => GatewayErrorKind::RateLimited,
        RemoteErrorKind::QuotaExceeded => GatewayErrorKind::QuotaExceeded,
        RemoteErrorKind::InvalidRequest => GatewayErrorKind::InvalidRequest,
        RemoteErrorKind::UnsupportedFeature => GatewayErrorKind::UnsupportedFeature,
        RemoteErrorKind::ModelNotFound => GatewayErrorKind::ModelNotFound,
        RemoteErrorKind::Timeout => GatewayErrorKind::Timeout,
        RemoteErrorKind::Overloaded => GatewayErrorKind::Overloaded,
        RemoteErrorKind::ServerRestarting => GatewayErrorKind::ServerRestarting,
        RemoteErrorKind::Transport => GatewayErrorKind::Transport,
        RemoteErrorKind::Remote => GatewayErrorKind::Internal,
    }
}

#[cfg(feature = "providers")]
fn provider_error_kind(kind: cogentlm_client::ProviderEndpointErrorKind) -> GatewayErrorKind {
    use cogentlm_client::ProviderEndpointErrorKind;
    match kind {
        ProviderEndpointErrorKind::Authentication => GatewayErrorKind::Authentication,
        ProviderEndpointErrorKind::Authorization => GatewayErrorKind::Authorization,
        ProviderEndpointErrorKind::RateLimited => GatewayErrorKind::RateLimited,
        ProviderEndpointErrorKind::QuotaExceeded => GatewayErrorKind::QuotaExceeded,
        ProviderEndpointErrorKind::InvalidRequest => GatewayErrorKind::InvalidRequest,
        ProviderEndpointErrorKind::UnsupportedFeature => GatewayErrorKind::UnsupportedFeature,
        ProviderEndpointErrorKind::ModelNotFound => GatewayErrorKind::ModelNotFound,
        ProviderEndpointErrorKind::Timeout => GatewayErrorKind::Timeout,
        ProviderEndpointErrorKind::Overloaded => GatewayErrorKind::Overloaded,
        ProviderEndpointErrorKind::Transport => GatewayErrorKind::Transport,
        ProviderEndpointErrorKind::Provider => GatewayErrorKind::Internal,
    }
}
