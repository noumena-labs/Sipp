use cogentlm_core::{FinishReason, TokenUsage};
use cogentlm_engine::engine::{PoolingType, RequestStats};

use crate::EndpointRef;

/// Final text response from a local or provider endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct CogentTextResponse {
    pub endpoint: EndpointRef,
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Option<TokenUsage>,
    pub local_stats: Option<RequestStats>,
}

/// Final embedding response from a local or provider endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct CogentEmbeddingResponse {
    pub endpoint: EndpointRef,
    pub values: Vec<f32>,
    pub usage: Option<TokenUsage>,
    pub local_stats: Option<RequestStats>,
    pub pooling: Option<PoolingType>,
    pub normalized: Option<bool>,
}
