use cogentlm_core::{FinishReason, TokenUsage};

use crate::{GatewayError, GatewayErrorKind, GatewayResult};

/// Normalized text output returned by a gateway text operation.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayTextOutput {
    /// Generated text.
    pub text: String,
    /// Normalized finish reason.
    pub finish_reason: FinishReason,
}

/// Normalized embedding output returned by a gateway embedding operation.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayEmbeddingOutput {
    /// Embedding vector.
    pub values: Vec<f32>,
}

/// Envelope shared by every gateway call.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayResponse<R> {
    /// Normalized operation result.
    pub result: R,
    /// Token usage when the gateway reports it.
    pub usage: Option<TokenUsage>,
    /// Gateway response metadata.
    pub metadata: GatewayResponseMetadata,
}

/// Text response returned by `query` or `chat`.
pub type GatewayTextResponse = GatewayResponse<GatewayTextOutput>;

/// Embedding response returned by `embed`.
pub type GatewayEmbeddingResponse = GatewayResponse<GatewayEmbeddingOutput>;

/// Metadata reported by the gateway for a normalized response.
#[derive(Debug, Clone, PartialEq)]
pub struct GatewayResponseMetadata {
    /// Public gateway alias.
    pub model: String,
    /// Gateway request id, usually from a response header.
    pub request_id: Option<String>,
    /// Gateway response id from the response body.
    pub response_id: Option<String>,
    /// Raw finish reason before normalization.
    pub finish_reason_raw: Option<String>,
    /// Raw gateway response body.
    pub raw: serde_json::Value,
}

pub(crate) fn text_response_from_body(
    request_id: Option<String>,
    body: serde_json::Value,
    redaction_secret: &str,
) -> GatewayResult<GatewayTextResponse> {
    reject_body_error(
        &body,
        "gateway text error",
        request_id.as_deref(),
        redaction_secret,
    )?;
    ensure_object_body(&body)?;
    let model = string_field(&body, "model")?;
    let text = string_field(&body, "text")?;
    let finish_reason_raw = string_field(&body, "finish_reason")?;
    let usage = body
        .get("usage")
        .filter(|value| !value.is_null())
        .map(token_usage)
        .transpose()?;

    Ok(GatewayTextResponse {
        result: GatewayTextOutput {
            text,
            finish_reason: map_finish_reason(Some(&finish_reason_raw)),
        },
        usage,
        metadata: GatewayResponseMetadata {
            model,
            request_id,
            response_id: body
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            finish_reason_raw: Some(finish_reason_raw),
            raw: body,
        },
    })
}

pub(crate) fn embedding_response_from_body(
    request_id: Option<String>,
    body: serde_json::Value,
    redaction_secret: &str,
) -> GatewayResult<GatewayEmbeddingResponse> {
    reject_body_error(
        &body,
        "gateway embedding error",
        request_id.as_deref(),
        redaction_secret,
    )?;
    ensure_object_body(&body)?;
    let model = string_field(&body, "model")?;
    let values = embedding_values(&body)?;
    let usage = body
        .get("usage")
        .filter(|value| !value.is_null())
        .map(token_usage)
        .transpose()?;

    Ok(GatewayEmbeddingResponse {
        result: GatewayEmbeddingOutput { values },
        usage,
        metadata: GatewayResponseMetadata {
            model,
            request_id,
            response_id: body
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            finish_reason_raw: None,
            raw: body,
        },
    })
}

pub(crate) fn map_finish_reason(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("length" | "max_tokens" | "max_output_tokens") => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

pub(crate) fn token_usage(value: &serde_json::Value) -> GatewayResult<TokenUsage> {
    if !value.is_object() {
        return Err(GatewayError::new(
            GatewayErrorKind::Gateway,
            "usage must be a JSON object",
        ));
    }
    Ok(TokenUsage {
        input_tokens: optional_u32(value, "input_tokens")?,
        output_tokens: optional_u32(value, "output_tokens")?,
        total_tokens: optional_u32(value, "total_tokens")?,
    })
}

pub(crate) fn gateway_body_error(
    raw: serde_json::Value,
    default_message: &'static str,
) -> GatewayError {
    let message = raw
        .pointer("/error/message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(default_message)
        .to_string();
    let code = raw
        .pointer("/error/code")
        .or_else(|| raw.pointer("/error/type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    GatewayError {
        kind: crate::error::gateway_error_kind_from_code(code.as_deref())
            .unwrap_or(GatewayErrorKind::Gateway),
        status: None,
        code,
        message,
        retry_after: None,
        request_id: None,
        raw: Some(Box::new(raw)),
    }
}

fn reject_body_error(
    body: &serde_json::Value,
    default_message: &'static str,
    request_id: Option<&str>,
    redaction_secret: &str,
) -> GatewayResult<()> {
    if body.get("error").is_some_and(|value| !value.is_null()) {
        let mut error = gateway_body_error(body.clone(), default_message);
        error.request_id = request_id.map(str::to_owned);
        error.redact_secret(redaction_secret);
        return Err(error);
    }
    Ok(())
}

fn ensure_object_body(body: &serde_json::Value) -> GatewayResult<()> {
    if body.is_object() {
        Ok(())
    } else {
        Err(GatewayError::new(
            GatewayErrorKind::Gateway,
            "gateway response must be a JSON object",
        ))
    }
}

fn string_field(body: &serde_json::Value, field: &'static str) -> GatewayResult<String> {
    body.get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            GatewayError::new(
                GatewayErrorKind::Gateway,
                format!("gateway response missing {field}"),
            )
        })
}

fn embedding_values(body: &serde_json::Value) -> GatewayResult<Vec<f32>> {
    let values = body
        .get("embedding")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            GatewayError::new(
                GatewayErrorKind::Gateway,
                "gateway embedding response missing vector",
            )
        })?;

    values.iter().map(embedding_value).collect()
}

fn embedding_value(value: &serde_json::Value) -> GatewayResult<f32> {
    let Some(value) = value.as_f64() else {
        return Err(GatewayError::new(
            GatewayErrorKind::Gateway,
            "embedding value is not numeric",
        ));
    };
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(GatewayError::new(
            GatewayErrorKind::Gateway,
            "embedding value is not representable as f32",
        ));
    }
    Ok(value as f32)
}

fn optional_u32(value: &serde_json::Value, key: &str) -> GatewayResult<Option<u32>> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    let Some(number) = raw.as_u64() else {
        return Err(GatewayError::new(
            GatewayErrorKind::Gateway,
            format!("usage field is not a number: {key}"),
        ));
    };
    u32::try_from(number).map(Some).map_err(|_| {
        GatewayError::new(
            GatewayErrorKind::Gateway,
            format!("usage field exceeds u32: {key}"),
        )
    })
}
