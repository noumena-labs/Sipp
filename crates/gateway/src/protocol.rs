use std::{collections::BTreeMap, pin::Pin};

use cogentlm_client::{
    CogentChatRequest, CogentEmbedRequest, CogentQueryRequest, CogentResponseMetadata,
    CogentTextOptions, EndpointRef,
};
use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenBatch, TokenUsage};
use futures_util::Stream;
use serde::{Deserialize, Serialize};

use crate::{GatewayError, GatewayResult};

/// Gateway-specific JSON options passed through request bodies.
pub type GatewayOptions = BTreeMap<String, serde_json::Value>;

/// Stream returned by gateway text operations.
pub type GatewayStream = Pin<Box<dyn Stream<Item = GatewayResult<GatewayStreamEvent>> + Send>>;

/// Gateway streaming event emitted by query and chat operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GatewayStreamEvent {
    /// Text token batch.
    TokenBatch(TokenBatch),
    /// Token usage.
    Usage { usage: TokenUsage },
    /// Final finish reason and execution metadata.
    Finished {
        /// Normalized finish reason.
        finish_reason: FinishReason,
        /// Correlation metadata from the client and upstream service.
        metadata: GatewayExecutionMetadata,
    },
}

/// Correlation metadata preserved through gateway execution.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct GatewayExecutionMetadata {
    /// Canonical gateway request ID.
    pub request_id: Option<String>,
    /// Upstream provider or gateway request ID.
    pub upstream_request_id: Option<String>,
    /// Upstream provider or gateway response ID.
    pub upstream_response_id: Option<String>,
}

impl From<CogentResponseMetadata> for GatewayExecutionMetadata {
    fn from(metadata: CogentResponseMetadata) -> Self {
        Self {
            request_id: metadata.request_id,
            upstream_request_id: metadata.upstream_request_id,
            upstream_response_id: metadata.upstream_response_id,
        }
    }
}

/// Normalized text output returned by a gateway executor.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayTextOutput {
    /// Generated text.
    pub text: String,
    /// Normalized finish reason.
    pub finish_reason: FinishReason,
    /// Token usage when available.
    pub usage: Option<TokenUsage>,
    /// Correlation metadata from the client and upstream service.
    pub metadata: GatewayExecutionMetadata,
}

/// Public query request body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QueryRequestBody {
    /// Public model alias.
    pub model: String,
    /// Raw prompt text.
    pub prompt: String,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff.
    pub top_p: Option<f32>,
    /// Stop strings.
    #[serde(default)]
    pub stop: Vec<String>,
    /// Whether to stream token events.
    #[serde(default)]
    pub stream: bool,
    /// Additional gateway options.
    #[serde(flatten)]
    pub gateway_options: GatewayOptions,
}

impl QueryRequestBody {
    pub(crate) fn into_client(self, endpoint: EndpointRef) -> CogentQueryRequest {
        CogentQueryRequest {
            endpoint: Some(endpoint),
            prompt: self.prompt,
            options: text_options(self.max_tokens, self.temperature, self.top_p, self.stop),
            gateway_options: self.gateway_options.into_iter().collect(),
            emit_tokens: self.stream,
            ..CogentQueryRequest::default()
        }
    }
}

/// Public chat request body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatRequestBody {
    /// Public model alias.
    pub model: String,
    /// Chat messages.
    pub messages: Vec<ChatMessageBody>,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff.
    pub top_p: Option<f32>,
    /// Stop strings.
    #[serde(default)]
    pub stop: Vec<String>,
    /// Whether to stream token events.
    #[serde(default)]
    pub stream: bool,
    /// Additional gateway options.
    #[serde(flatten)]
    pub gateway_options: GatewayOptions,
}

impl ChatRequestBody {
    pub(crate) fn into_client(self, endpoint: EndpointRef) -> CogentChatRequest {
        CogentChatRequest {
            endpoint: Some(endpoint),
            messages: self
                .messages
                .into_iter()
                .map(ChatMessageBody::into_core)
                .collect(),
            options: text_options(self.max_tokens, self.temperature, self.top_p, self.stop),
            gateway_options: self.gateway_options.into_iter().collect(),
            emit_tokens: self.stream,
            ..CogentChatRequest::default()
        }
    }
}

/// Public embedding request body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbedRequestBody {
    /// Public model alias.
    pub model: String,
    /// Input text to embed.
    pub input: String,
    /// Additional gateway options.
    #[serde(flatten)]
    pub gateway_options: GatewayOptions,
}

impl EmbedRequestBody {
    pub(crate) fn into_client(self, endpoint: EndpointRef) -> CogentEmbedRequest {
        CogentEmbedRequest {
            endpoint: Some(endpoint),
            input: self.input,
            gateway_options: self.gateway_options.into_iter().collect(),
            ..CogentEmbedRequest::default()
        }
    }
}

/// Chat message accepted by the gateway protocol.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessageBody {
    /// Chat role.
    pub role: ChatRole,
    /// Message content.
    pub content: String,
}

impl ChatMessageBody {
    fn into_core(self) -> ChatMessage {
        ChatMessage::new(self.role, self.content)
    }
}

/// Public finite text response.
#[derive(Debug, Clone, Serialize)]
pub struct TextResponseBody {
    /// Gateway response ID.
    pub id: String,
    /// Public model alias.
    pub model: String,
    /// Generated text.
    pub text: String,
    /// Normalized finish reason.
    pub finish_reason: String,
    /// Usage when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageBody>,
}

/// Public finite embedding response.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponseBody {
    /// Gateway response ID.
    pub id: String,
    /// Public model alias.
    pub model: String,
    /// Embedding vector.
    pub embedding: Vec<f32>,
    /// Usage when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageBody>,
}

/// Public token usage.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct UsageBody {
    /// Input token count when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    /// Output token count when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    /// Total token count when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

impl From<TokenUsage> for UsageBody {
    fn from(usage: TokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

/// Public error envelope.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    /// Error payload.
    pub error: ErrorBody,
}

impl From<&GatewayError> for ErrorEnvelope {
    fn from(error: &GatewayError) -> Self {
        Self {
            error: ErrorBody {
                code: error.code(),
                message: error.message.clone(),
            },
        }
    }
}

/// Public error body.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    /// Stable machine-readable error code.
    pub code: &'static str,
    /// Client-safe message.
    pub message: String,
}

/// Convert a finish reason into its stable wire label.
pub fn finish_reason(reason: FinishReason) -> String {
    reason.as_str().to_string()
}

fn text_options(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Vec<String>,
) -> CogentTextOptions {
    CogentTextOptions {
        max_tokens,
        temperature,
        top_p,
        stop,
    }
}
