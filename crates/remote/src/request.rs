use cogentlm_core::ChatMessage;

use crate::{GatewayError, GatewayErrorKind, GatewayResult};

/// Gateway-specific free-form options carried by request envelopes.
pub type GatewayOptions = serde_json::Map<String, serde_json::Value>;

/// Text generation options shared by gateway text operations.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GatewayGenerationOptions {
    /// Maximum output tokens requested from the gateway alias.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff.
    pub top_p: Option<f32>,
    /// Stop strings.
    pub stop: Vec<String>,
}

/// Gateway raw-prompt request.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayQueryRequest {
    /// Public gateway alias.
    pub model: String,
    /// Raw prompt text.
    pub prompt: String,
    /// Shared generation options.
    pub options: GatewayGenerationOptions,
    /// Gateway-specific options.
    pub gateway_options: GatewayOptions,
}

/// Gateway chat request.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayChatRequest {
    /// Public gateway alias.
    pub model: String,
    /// Chat messages.
    pub messages: Vec<ChatMessage>,
    /// Shared generation options.
    pub options: GatewayGenerationOptions,
    /// Gateway-specific options.
    pub gateway_options: GatewayOptions,
}

/// Gateway embedding request.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayEmbedRequest {
    /// Public gateway alias.
    pub model: String,
    /// Input text to embed.
    pub input: String,
    /// Gateway-specific options.
    pub gateway_options: GatewayOptions,
}

const QUERY_TYPED_FIELDS: &[&str] = &[
    "model",
    "prompt",
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "stream",
];
const CHAT_TYPED_FIELDS: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "stream",
];
const EMBED_TYPED_FIELDS: &[&str] = &["model", "input"];
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

pub(crate) fn query_body(
    req: &GatewayQueryRequest,
    stream: bool,
) -> GatewayResult<serde_json::Value> {
    require_non_empty_field(&req.model, "model")?;

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::Value::String(req.model.clone()),
    );
    body.insert(
        "prompt".to_string(),
        serde_json::Value::String(req.prompt.clone()),
    );
    insert_generation_options(&mut body, &req.options)?;
    body.insert("stream".to_string(), serde_json::json!(stream));
    merge_gateway_options(&mut body, &req.gateway_options, QUERY_TYPED_FIELDS)?;
    Ok(serde_json::Value::Object(body))
}

pub(crate) fn chat_body(
    req: &GatewayChatRequest,
    stream: bool,
) -> GatewayResult<serde_json::Value> {
    require_non_empty_field(&req.model, "model")?;
    if req.messages.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "gateway chat messages must not be empty",
        ));
    }

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::Value::String(req.model.clone()),
    );
    body.insert(
        "messages".to_string(),
        serde_json::Value::Array(req.messages.iter().map(gateway_message).collect()),
    );
    insert_generation_options(&mut body, &req.options)?;
    body.insert("stream".to_string(), serde_json::json!(stream));
    merge_gateway_options(&mut body, &req.gateway_options, CHAT_TYPED_FIELDS)?;
    Ok(serde_json::Value::Object(body))
}

pub(crate) fn embed_body(req: &GatewayEmbedRequest) -> GatewayResult<serde_json::Value> {
    require_non_empty_field(&req.model, "model")?;
    require_non_empty_field(&req.input, "input")?;

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::Value::String(req.model.clone()),
    );
    body.insert(
        "input".to_string(),
        serde_json::Value::String(req.input.clone()),
    );
    merge_gateway_options(&mut body, &req.gateway_options, EMBED_TYPED_FIELDS)?;
    Ok(serde_json::Value::Object(body))
}

fn insert_generation_options(
    body: &mut serde_json::Map<String, serde_json::Value>,
    options: &GatewayGenerationOptions,
) -> GatewayResult<()> {
    insert_positive_u32_option(body, "max_tokens", options.max_tokens)?;
    insert_finite_f32_option(body, "temperature", options.temperature)?;
    insert_finite_f32_option(body, "top_p", options.top_p)?;
    if !options.stop.is_empty() {
        body.insert("stop".to_string(), serde_json::json!(options.stop));
    }
    Ok(())
}

fn gateway_message(message: &ChatMessage) -> serde_json::Value {
    serde_json::json!({
        "role": message.role.as_str(),
        "content": message.content,
    })
}

fn require_non_empty_field(value: &str, field: &'static str) -> GatewayResult<()> {
    if value.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not be empty"),
        ));
    }
    Ok(())
}

fn insert_positive_u32_option(
    body: &mut serde_json::Map<String, serde_json::Value>,
    key: &'static str,
    value: Option<u32>,
) -> GatewayResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value == 0 {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{key} must be greater than zero"),
        ));
    }
    body.insert(key.to_string(), serde_json::json!(value));
    Ok(())
}

fn insert_finite_f32_option(
    body: &mut serde_json::Map<String, serde_json::Value>,
    key: &'static str,
    value: Option<f32>,
) -> GatewayResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{key} must be finite"),
        ));
    }
    body.insert(key.to_string(), serde_json::json!(value));
    Ok(())
}

fn merge_gateway_options(
    body: &mut serde_json::Map<String, serde_json::Value>,
    gateway_options: &GatewayOptions,
    typed_fields: &[&str],
) -> GatewayResult<()> {
    for (key, value) in gateway_options {
        if typed_fields.contains(&key.as_str()) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("gateway_options cannot override typed field: {key}"),
            ));
        }
        if LOCAL_ONLY_GATEWAY_FIELDS.contains(&key.as_str()) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("gateway_options cannot contain local-only field: {key}"),
            ));
        }
        body.insert(key.clone(), value.clone());
    }
    Ok(())
}
