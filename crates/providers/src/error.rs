use std::time::Duration;

use thiserror::Error;

use crate::ProviderKind;

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

#[derive(Debug, Error)]
#[error("{provider:?} provider error ({kind:?}): {message}")]
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
        Some("authentication_error") => Some(ProviderErrorKind::Authentication),
        Some("permission_error") => Some(ProviderErrorKind::Authorization),
        Some("invalid_request_error") => Some(ProviderErrorKind::InvalidRequest),
        Some("not_found_error") => Some(ProviderErrorKind::ModelNotFound),
        Some("overloaded_error") => Some(ProviderErrorKind::Overloaded),
        Some("insufficient_quota" | "quota_exceeded") => Some(ProviderErrorKind::QuotaExceeded),
        Some("rate_limit" | "rate_limited" | "rate_limit_exceeded" | "rate_limit_error") => {
            Some(ProviderErrorKind::RateLimited)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_kind_from_code_maps_retry_relevant_codes() {
        assert_eq!(
            provider_error_kind_from_code(Some("rate_limit")),
            Some(ProviderErrorKind::RateLimited)
        );
        assert_eq!(
            provider_error_kind_from_code(Some("insufficient_quota")),
            Some(ProviderErrorKind::QuotaExceeded)
        );
        assert_eq!(
            provider_error_kind_from_code(Some("overloaded_error")),
            Some(ProviderErrorKind::Overloaded)
        );
        assert_eq!(provider_error_kind_from_code(Some("bad_request")), None);
    }
}
