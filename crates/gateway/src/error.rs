use std::time::Duration;

use axum::{
    http::{header::HeaderName, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
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

    pub(crate) const fn status_code(self) -> StatusCode {
        match self {
            Self::Authentication => StatusCode::UNAUTHORIZED,
            Self::Authorization => StatusCode::FORBIDDEN,
            Self::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            Self::QuotaExceeded => StatusCode::PAYMENT_REQUIRED,
            Self::InvalidRequest | Self::UnsupportedFeature => StatusCode::BAD_REQUEST,
            Self::RequestTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::ModelNotFound => StatusCode::NOT_FOUND,
            Self::Timeout => StatusCode::REQUEST_TIMEOUT,
            Self::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
            Self::Transport | Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Normalized gateway error.
#[derive(Debug, Clone, Error)]
#[error("gateway error ({}): {message}", kind.as_str())]
pub struct GatewayError {
    /// Error classification.
    pub kind: GatewayErrorKind,
    /// Public gateway error code.
    pub code: String,
    /// Human-readable message safe to return to clients.
    pub message: String,
    /// Retry delay when applicable.
    pub retry_after: Option<Duration>,
    /// Upstream or gateway request id when available.
    pub request_id: Option<String>,
}

impl GatewayError {
    /// Create a normalized gateway error.
    pub fn new(kind: GatewayErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            code: kind.as_str().to_string(),
            message: message.into(),
            retry_after: None,
            request_id: None,
        }
    }

    pub(crate) fn with_retry_after(mut self, retry_after: Option<Duration>) -> Self {
        self.retry_after = retry_after;
        self
    }

    pub(crate) fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        if let Some(retry_after) = self.retry_after {
            insert_header_if_valid(
                &mut headers,
                HeaderName::from_static("retry-after"),
                retry_after.as_secs().to_string(),
            );
            insert_header_if_valid(
                &mut headers,
                HeaderName::from_static("retry-after-ms"),
                retry_after.as_millis().to_string(),
            );
        }
        if let Some(request_id) = &self.request_id {
            insert_header_if_valid(
                &mut headers,
                HeaderName::from_static("x-request-id"),
                request_id,
            );
        }

        let status = self.kind.status_code();
        let body = Json(ErrorEnvelope {
            error: ErrorBody {
                code: self.code,
                message: self.message,
            },
        });
        (status, headers, body).into_response()
    }
}

fn insert_header_if_valid(headers: &mut HeaderMap, name: HeaderName, value: impl AsRef<str>) {
    if let Ok(value) = HeaderValue::from_str(value.as_ref()) {
        headers.insert(name, value);
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: String,
    message: String,
}
