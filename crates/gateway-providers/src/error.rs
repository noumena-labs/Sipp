use std::{fmt, time::Duration};

use thiserror::Error;

use crate::ProviderKind;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "tests/error_tests.rs"]
mod error_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub type ProviderResult<T> = Result<T, ProviderError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    Authentication,
    Authorization,
    RateLimited,
    QuotaExceeded,
    InvalidRequest,
    UnsupportedFeature,
    ModelNotFound,
    Timeout,
    Overloaded,
    Transport,
    Provider,
}

impl ProviderErrorKind {
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

#[derive(Error)]
#[error("{provider:?} provider error ({kind:?})")]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub provider: ProviderKind,
    pub status: Option<u16>,
    pub code: Option<String>,
    pub message: String,
    pub retry_after: Option<Duration>,
    pub request_id: Option<String>,
    pub raw: Option<Box<serde_json::Value>>,
}

impl fmt::Debug for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderError")
            .field("kind", &self.kind)
            .field("provider", &self.provider)
            .field("status", &self.status)
            .field("code", &self.code.as_ref().map(|_| "[redacted]"))
            .field("message", &"[redacted]")
            .field("retry_after", &self.retry_after)
            .field(
                "request_id",
                &self.request_id.as_ref().map(|_| "[redacted]"),
            )
            .field("raw", &self.raw.as_ref().map(|_| "[redacted]"))
            .finish()
    }
}

impl ProviderError {
    pub fn new(
        kind: ProviderErrorKind,
        provider: ProviderKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            provider,
            status: None,
            code: None,
            message: message.into(),
            retry_after: None,
            request_id: None,
            raw: None,
        }
    }
}

pub(crate) fn provider_error_kind_from_code(code: Option<&str>) -> Option<ProviderErrorKind> {
    match code {
        Some("authentication" | "authentication_error") => Some(ProviderErrorKind::Authentication),
        Some("authorization" | "authorization_error" | "permission_error") => {
            Some(ProviderErrorKind::Authorization)
        }
        Some("invalid_request" | "invalid_request_error") => {
            Some(ProviderErrorKind::InvalidRequest)
        }
        Some("unsupported_feature") => Some(ProviderErrorKind::UnsupportedFeature),
        Some("model_not_found" | "not_found_error") => Some(ProviderErrorKind::ModelNotFound),
        Some("overloaded" | "overloaded_error") => Some(ProviderErrorKind::Overloaded),
        Some("insufficient_quota" | "quota_exceeded") => Some(ProviderErrorKind::QuotaExceeded),
        Some("rate_limit" | "rate_limited" | "rate_limit_exceeded" | "rate_limit_error") => {
            Some(ProviderErrorKind::RateLimited)
        }
        Some("timeout") => Some(ProviderErrorKind::Timeout),
        Some("transport") => Some(ProviderErrorKind::Transport),
        _ => None,
    }
}
