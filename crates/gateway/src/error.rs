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
            Self::Timeout => 408,
            Self::Overloaded => 503,
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
}

impl GatewayError {
    /// Create a normalized gateway error.
    pub fn new(kind: GatewayErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retry_after: None,
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
}
