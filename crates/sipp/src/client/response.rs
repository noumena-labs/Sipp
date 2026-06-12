use crate::core::{FinishReason, TokenUsage};
use crate::engine::{PoolingType, RequestStats};

use crate::client::EndpointRef;

/// Correlation metadata returned by local, gateway, and provider endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SippResponseMetadata {
    /// Canonical request identifier supplied by the calling application.
    pub request_id: Option<String>,
    /// Request identifier returned by an upstream gateway or provider.
    pub upstream_request_id: Option<String>,
    /// Response identifier returned by an upstream gateway or provider.
    pub upstream_response_id: Option<String>,
}

/// Final text response from an inference endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct SippTextResponse {
    /// Endpoint that produced the response.
    pub endpoint: EndpointRef,
    /// Generated text.
    pub text: String,
    /// Completion finish reason.
    pub finish_reason: FinishReason,
    /// Token usage when reported by the endpoint.
    pub usage: Option<TokenUsage>,
    /// Local runtime statistics for local endpoints.
    pub local_stats: Option<RequestStats>,
    /// Request and upstream correlation metadata.
    pub metadata: SippResponseMetadata,
}

/// Final embedding response from an inference endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct SippEmbeddingResponse {
    /// Endpoint that produced the response.
    pub endpoint: EndpointRef,
    /// Embedding vector.
    pub values: Vec<f32>,
    /// Token usage when reported by the endpoint.
    pub usage: Option<TokenUsage>,
    /// Local runtime statistics for local endpoints.
    pub local_stats: Option<RequestStats>,
    /// Pooling used by local embedding models.
    pub pooling: Option<PoolingType>,
    /// Whether the endpoint normalized the vector.
    pub normalized: Option<bool>,
    /// Request and upstream correlation metadata.
    pub metadata: SippResponseMetadata,
}
