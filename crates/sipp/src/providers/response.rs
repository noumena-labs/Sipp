use crate::core::{FinishReason, TokenUsage};

use crate::providers::ProviderKind;

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderTextOutput {
    pub text: String,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderEmbeddingOutput {
    pub values: Vec<f32>,
}

/// Envelope shared by every provider call: the normalized `result` plus
/// optional usage and provider metadata. `R` is the call's output type.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderResponse<R> {
    pub result: R,
    pub usage: Option<TokenUsage>,
    pub metadata: ProviderResponseMetadata,
}

pub type ProviderChatResponse = ProviderResponse<ProviderTextOutput>;
pub type ProviderGenerateResponse = ProviderResponse<ProviderTextOutput>;
pub type ProviderEmbeddingResponse = ProviderResponse<ProviderEmbeddingOutput>;

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderResponseMetadata {
    pub provider: ProviderKind,
    pub model: String,
    pub request_id: Option<String>,
    pub response_id: Option<String>,
    pub finish_reason_raw: Option<String>,
    pub raw: serde_json::Value,
}
