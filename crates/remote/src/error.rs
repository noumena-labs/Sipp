use std::time::Duration;

use thiserror::Error;

/// Result type returned by gateway transport operations.
pub type GatewayResult<T> = Result<T, GatewayError>;

/// Classification for CogentLM gateway transport and API failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayErrorKind {
    /// Authentication failed.
    Authentication,
    /// Authorization failed after authentication.
    Authorization,
    /// The gateway rate limited the request.
    RateLimited,
    /// The gateway account or tenant quota is exhausted.
    QuotaExceeded,
    /// The request is invalid for the gateway operation.
    InvalidRequest,
    /// The selected alias does not support the requested operation.
    UnsupportedFeature,
    /// The requested public alias was not found.
    ModelNotFound,
    /// The remote request timed out.
    Timeout,
    /// The gateway or upstream backend is overloaded.
    Overloaded,
    /// Network or protocol transport failed.
    Transport,
    /// Gateway returned an unclassified error.
    Gateway,
}

impl GatewayErrorKind {
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
            Self::Gateway => "gateway",
        }
    }
}

/// Structured error returned by a CogentLM remote gateway.
#[derive(Debug, Clone, Error)]
#[error("gateway error ({}): {message}", kind.as_str())]
pub struct GatewayError {
    /// Error classification.
    pub kind: GatewayErrorKind,
    /// HTTP status code when the gateway returned one.
    pub status: Option<u16>,
    /// Gateway error code when available.
    pub code: Option<String>,
    /// Human-readable error message.
    pub message: String,
    /// Retry delay returned by the gateway.
    pub retry_after: Option<Duration>,
    /// Gateway request id when available.
    pub request_id: Option<String>,
    /// Raw gateway error payload when available.
    pub raw: Option<Box<serde_json::Value>>,
}

impl GatewayError {
    /// Create a gateway error with no optional transport metadata.
    pub fn new(kind: GatewayErrorKind, message: impl Into<String>) -> Self {
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

    pub(crate) fn redact_secret(&mut self, secret: &str) {
        if secret.is_empty() {
            return;
        }
        self.message = redact_string(&self.message, secret);
        self.code = self.code.as_ref().map(|code| redact_string(code, secret));
        self.request_id = self
            .request_id
            .as_ref()
            .map(|request_id| redact_string(request_id, secret));
        if let Some(raw) = &mut self.raw {
            redact_json_value(raw, secret);
        }
    }
}

fn redact_string(value: &str, secret: &str) -> String {
    value.replace(secret, "[redacted]")
}

fn redact_json_value(value: &mut serde_json::Value, secret: &str) {
    match value {
        serde_json::Value::String(text) => {
            *text = redact_string(text, secret);
        }
        serde_json::Value::Array(items) => {
            for item in items {
                redact_json_value(item, secret);
            }
        }
        serde_json::Value::Object(fields) => {
            for value in fields.values_mut() {
                redact_json_value(value, secret);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

pub(crate) fn gateway_error_kind_from_code(code: Option<&str>) -> Option<GatewayErrorKind> {
    match code {
        Some("authentication" | "authentication_error") => Some(GatewayErrorKind::Authentication),
        Some("authorization" | "authorization_error" | "permission_error") => {
            Some(GatewayErrorKind::Authorization)
        }
        Some("invalid_request" | "invalid_request_error") => Some(GatewayErrorKind::InvalidRequest),
        Some("unsupported_feature") => Some(GatewayErrorKind::UnsupportedFeature),
        Some("model_not_found" | "not_found_error") => Some(GatewayErrorKind::ModelNotFound),
        Some("overloaded" | "overloaded_error") => Some(GatewayErrorKind::Overloaded),
        Some("insufficient_quota" | "quota_exceeded") => Some(GatewayErrorKind::QuotaExceeded),
        Some("rate_limit" | "rate_limited" | "rate_limit_exceeded" | "rate_limit_error") => {
            Some(GatewayErrorKind::RateLimited)
        }
        Some("timeout") => Some(GatewayErrorKind::Timeout),
        Some("transport") => Some(GatewayErrorKind::Transport),
        _ => None,
    }
}
