use std::collections::BTreeMap;

use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenUsage};
use serde::{Deserialize, Serialize};

use crate::{
    BackendChatRequest, BackendEmbedRequest, BackendGenerationOptions, BackendQueryRequest,
    GatewayError, GatewayErrorKind,
};

pub(crate) type GatewayOptions = BTreeMap<String, serde_json::Value>;
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
pub(crate) struct QueryRequestBody {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub gateway_options: GatewayOptions,
}

impl QueryRequestBody {
    pub(crate) fn into_backend(self) -> BackendQueryRequest {
        let options = self.generation_options();
        BackendQueryRequest {
            prompt: self.prompt,
            options,
            gateway_options: self.gateway_options,
        }
    }

    fn generation_options(&self) -> BackendGenerationOptions {
        BackendGenerationOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatRequestBody {
    pub model: String,
    pub messages: Vec<ChatMessageBody>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub gateway_options: GatewayOptions,
}

impl ChatRequestBody {
    pub(crate) fn into_backend(self) -> BackendChatRequest {
        BackendChatRequest {
            messages: self
                .messages
                .into_iter()
                .map(ChatMessageBody::into_core)
                .collect(),
            options: BackendGenerationOptions {
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                stop: self.stop,
            },
            gateway_options: self.gateway_options,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EmbedRequestBody {
    pub model: String,
    pub input: String,
    #[serde(flatten)]
    pub gateway_options: GatewayOptions,
}

impl EmbedRequestBody {
    pub(crate) fn into_backend(self) -> BackendEmbedRequest {
        BackendEmbedRequest {
            input: self.input,
            gateway_options: self.gateway_options,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatMessageBody {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessageBody {
    fn into_core(self) -> ChatMessage {
        ChatMessage::new(self.role, self.content)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TextResponseBody {
    pub id: String,
    pub model: String,
    pub text: String,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageBody>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EmbeddingResponseBody {
    pub id: String,
    pub model: String,
    pub embedding: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageBody>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct UsageBody {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
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

pub(crate) fn finish_reason(reason: FinishReason) -> String {
    reason.as_str().to_string()
}

pub(crate) fn validate_text_options(
    options: &BackendGenerationOptions,
) -> Result<(), GatewayError> {
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

pub(crate) fn validate_gateway_options(options: &GatewayOptions) -> Result<(), GatewayError> {
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

pub(crate) fn validate_non_empty(value: &str, field: &'static str) -> Result<(), GatewayError> {
    if value.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not be empty"),
        ));
    }
    Ok(())
}
