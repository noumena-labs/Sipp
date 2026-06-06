use std::collections::BTreeMap;

use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenUsage};
use serde::{Deserialize, Serialize};

use crate::{
    BackendChatRequest, BackendEmbedRequest, BackendGenerationOptions, BackendQueryRequest,
    GatewayError, GatewayErrorKind,
};

/// Gateway-specific JSON options passed through request bodies.
pub type GatewayOptions = BTreeMap<String, serde_json::Value>;
const LOCAL_ONLY_GATEWAY_FIELDS: &[&str] = &[
    "context_key",
    "contextKey",
    "session",
    "grammar",
    "json_schema",
    "jsonSchema",
    "sampling",
    "media",
    "normalize",
    "local",
];

#[derive(Debug, Clone, Deserialize)]
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
    /// Convert the body into a backend request.
    pub fn into_backend(self) -> BackendQueryRequest {
        let options = self.generation_options();
        BackendQueryRequest {
            prompt: self.prompt,
            options,
            gateway_options: self.gateway_options,
        }
    }

    /// Return normalized generation options.
    pub fn generation_options(&self) -> BackendGenerationOptions {
        BackendGenerationOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
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
    /// Convert the body into a backend request.
    pub fn into_backend(self) -> BackendChatRequest {
        let options = self.generation_options();
        BackendChatRequest {
            messages: self
                .messages
                .into_iter()
                .map(ChatMessageBody::into_core)
                .collect(),
            options,
            gateway_options: self.gateway_options,
        }
    }

    /// Return normalized generation options.
    pub fn generation_options(&self) -> BackendGenerationOptions {
        BackendGenerationOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
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
    /// Convert the body into a backend request.
    pub fn into_backend(self) -> BackendEmbedRequest {
        BackendEmbedRequest {
            input: self.input,
            gateway_options: self.gateway_options,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Serialize)]
pub struct TextResponseBody {
    /// Response ID.
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

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponseBody {
    /// Response ID.
    pub id: String,
    /// Public model alias.
    pub model: String,
    /// Embedding vector.
    pub embedding: Vec<f32>,
    /// Usage when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageBody>,
}

#[derive(Debug, Clone, Serialize)]
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

/// Convert a finish reason into its stable wire label.
pub fn finish_reason(reason: FinishReason) -> String {
    reason.as_str().to_string()
}

/// Validate text generation options.
pub fn validate_text_options(options: &BackendGenerationOptions) -> Result<(), GatewayError> {
    if matches!(options.max_tokens, Some(0)) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "max_tokens must be greater than zero",
        ));
    }
    if options.temperature.is_some_and(|value| !value.is_finite()) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "temperature must be finite",
        ));
    }
    if options.temperature.is_some_and(|value| value < 0.0) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "temperature must be greater than or equal to zero",
        ));
    }
    if options.top_p.is_some_and(|value| !value.is_finite()) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "top_p must be finite",
        ));
    }
    if options
        .top_p
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "top_p must be between 0 and 1",
        ));
    }
    Ok(())
}

/// Validate gateway options for remote gateway requests.
pub fn validate_gateway_options(options: &GatewayOptions) -> Result<(), GatewayError> {
    for key in options.keys() {
        if LOCAL_ONLY_GATEWAY_FIELDS.contains(&key.as_str()) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("gateway request cannot contain local-only field: {key}"),
            ));
        }
    }
    Ok(())
}

/// Validate that a request string is not blank.
pub fn validate_non_empty(value: &str, field: &'static str) -> Result<(), GatewayError> {
    if value.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not be empty"),
        ));
    }
    Ok(())
}
