use cogentlm_core::{FinishReason, TokenUsage};
use cogentlm_engine::engine::{PoolingType, RequestStats};

use crate::EndpointRef;

/// Final text response from a local or remote endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct CogentTextResponse {
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
}

/// Final embedding response from a local or remote endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct CogentEmbeddingResponse {
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
}
