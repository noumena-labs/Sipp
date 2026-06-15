use std::collections::BTreeMap;

use bytes::Bytes;
use http::{HeaderMap, StatusCode};
use sipp::gateway_core::{GatewayError, GatewayErrorKind, GatewayRequestContext};
use sipp::{
    SippChatRequest, SippEmbedRequest, SippEmbeddingResponse, SippQueryRequest, SippTextResponse,
};
use thiserror::Error;

/// Result returned by gateway HTTP helper extension points.
pub type ToolkitResult<T> = Result<T, GatewayHttpError>;

/// A decoded protocol request and its application target.
pub struct DecodedRequest<T> {
    /// Application target selected by the incoming request.
    pub target: String,
    /// Whether the protocol requested streaming.
    pub stream: bool,
    /// Typed inference request.
    pub request: T,
}

/// Application-defined authentication output.
#[derive(Debug, Clone, Default)]
pub struct AuthenticatedRequest {
    /// Metadata made available to route and policy implementations.
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Decode and encode an application wire protocol.
pub trait ProtocolCodec: Send + Sync {
    /// Decode a query request.
    fn decode_query(&self, body: &[u8]) -> ToolkitResult<DecodedRequest<SippQueryRequest>>;
    /// Decode a chat request.
    fn decode_chat(&self, body: &[u8]) -> ToolkitResult<DecodedRequest<SippChatRequest>>;
    /// Decode an embedding request.
    fn decode_embed(&self, body: &[u8]) -> ToolkitResult<DecodedRequest<SippEmbedRequest>>;
    /// Encode a finite text response.
    fn encode_text(&self, target: &str, response: &SippTextResponse) -> ToolkitResult<Bytes>;
    /// Encode a finite embedding response.
    fn encode_embedding(
        &self,
        target: &str,
        response: &SippEmbeddingResponse,
    ) -> ToolkitResult<Bytes>;
    /// Encode a streaming event.
    fn encode_stream_event(
        &self,
        event: &sipp::gateway_core::GatewayStreamEvent,
    ) -> ToolkitResult<Bytes>;
    /// Encode an error after a streaming response has started.
    fn encode_stream_error(&self, error: &GatewayHttpError) -> Bytes;
    /// Encode an application error.
    fn encode_error(&self, error: &GatewayHttpError) -> Bytes;
    /// Return the response content type.
    fn content_type(&self, streaming: bool) -> &'static str;
}

/// Authenticate an incoming request without prescribing an auth scheme.
pub trait Authenticator: Send + Sync {
    /// Authenticate headers and return policy metadata.
    fn authenticate(&self, headers: &HeaderMap) -> ToolkitResult<AuthenticatedRequest>;
}

/// Translate protocol-neutral pipeline errors into application HTTP errors.
pub trait ErrorTranslator: Send + Sync {
    /// Translate a core error.
    fn translate(&self, error: GatewayError) -> GatewayHttpError;
}

/// Authenticator that accepts every request without metadata.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoAuthentication;

impl Authenticator for NoAuthentication {
    fn authenticate(&self, _headers: &HeaderMap) -> ToolkitResult<AuthenticatedRequest> {
        Ok(AuthenticatedRequest::default())
    }
}

/// Default protocol-neutral gateway error translator.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultErrorTranslator;

impl ErrorTranslator for DefaultErrorTranslator {
    fn translate(&self, error: GatewayError) -> GatewayHttpError {
        GatewayHttpError::from_gateway_error(error)
    }
}

/// Application HTTP error selected by auth, codecs, or error translation.
#[derive(Debug, Clone, Error)]
#[error("{code}: {message}")]
pub struct GatewayHttpError {
    /// HTTP response status.
    pub status: StatusCode,
    /// Stable application error code.
    pub code: String,
    /// Client-facing message.
    pub message: String,
}

impl GatewayHttpError {
    /// Create a bad-request error.
    pub fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, code, message)
    }

    /// Create an internal-server error.
    pub fn internal(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, code, message)
    }

    /// Translate a protocol-neutral gateway error.
    pub fn from_gateway_error(error: GatewayError) -> Self {
        let status = match error.kind {
            GatewayErrorKind::Resolution => StatusCode::NOT_FOUND,
            GatewayErrorKind::Authorization => StatusCode::FORBIDDEN,
            GatewayErrorKind::Admission => StatusCode::TOO_MANY_REQUESTS,
            GatewayErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
            GatewayErrorKind::Cancelled => StatusCode::REQUEST_TIMEOUT,
            GatewayErrorKind::Execution | GatewayErrorKind::Internal => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        Self::new(
            status,
            format!("{:?}", error.kind).to_ascii_lowercase(),
            error.message,
        )
    }

    /// Create an HTTP error.
    pub fn new(status: StatusCode, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Return the request id supplied by an HTTP boundary.
pub fn request_id(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
}

/// Build a gateway request context from route-owned authentication output.
pub fn request_context(
    request_id: Option<&str>,
    authenticated: AuthenticatedRequest,
) -> GatewayRequestContext {
    GatewayRequestContext::new(request_id.map(str::to_string)).with_metadata(authenticated.metadata)
}
