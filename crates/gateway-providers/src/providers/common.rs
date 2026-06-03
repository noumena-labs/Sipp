use crate::error::provider_error_kind_from_code;
use crate::{ProviderError, ProviderErrorKind, ProviderKind, ProviderOptions, ProviderResult};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
pub(super) fn require_non_empty_field(
    value: &str,
    field: &'static str,
    provider: ProviderKind,
) -> ProviderResult<()> {
    if value.trim().is_empty() {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            format!("{field} must not be empty"),
        ));
    }
    Ok(())
}

pub(super) fn insert_positive_u32_option(
    body: &mut serde_json::Map<String, serde_json::Value>,
    key: &'static str,
    value: Option<u32>,
    provider: ProviderKind,
) -> ProviderResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value == 0 {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            format!("{key} must be greater than zero"),
        ));
    }
    body.insert(key.to_string(), serde_json::json!(value));
    Ok(())
}

pub(super) fn insert_finite_f32_option(
    body: &mut serde_json::Map<String, serde_json::Value>,
    key: &'static str,
    value: Option<f32>,
    provider: ProviderKind,
) -> ProviderResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            format!("{key} must be finite"),
        ));
    }
    if key == "temperature" && value < 0.0 {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            "temperature must be greater than or equal to zero",
        ));
    }
    if key == "top_p" && !(0.0..=1.0).contains(&value) {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            "top_p must be between 0 and 1",
        ));
    }
    body.insert(key.to_string(), serde_json::json!(value));
    Ok(())
}

pub(super) fn merge_provider_options(
    body: &mut serde_json::Map<String, serde_json::Value>,
    provider_options: &ProviderOptions,
    typed_fields: &[&str],
    provider: ProviderKind,
) -> ProviderResult<()> {
    for (key, value) in provider_options {
        if typed_fields.contains(&key.as_str()) {
            return Err(ProviderError::new(
                ProviderErrorKind::InvalidRequest,
                provider,
                format!("provider_options cannot override typed field: {key}"),
            ));
        }
        body.insert(key.clone(), value.clone());
    }
    Ok(())
}

pub(super) fn optional_u32(
    value: &serde_json::Value,
    key: &str,
    provider: ProviderKind,
) -> ProviderResult<Option<u32>> {
    if !value.is_object() {
        return Err(provider_response_error(
            "usage must be a JSON object",
            provider,
        ));
    }
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    let Some(number) = raw.as_u64() else {
        return Err(provider_response_error(
            format!("usage field is not a number: {key}"),
            provider,
        ));
    };
    u32::try_from(number)
        .map(Some)
        .map_err(|_| provider_response_error(format!("usage field exceeds u32: {key}"), provider))
}

pub(super) fn token_usage_total(
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
) -> Option<u32> {
    input_tokens?.checked_add(output_tokens?)
}

pub(super) fn provider_response_error(
    message: impl Into<String>,
    provider: ProviderKind,
) -> ProviderError {
    ProviderError::new(ProviderErrorKind::Provider, provider, message.into())
}

pub(super) fn provider_body_error(
    raw: serde_json::Value,
    provider: ProviderKind,
    default_message: &'static str,
) -> ProviderError {
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

    ProviderError {
        kind: provider_error_kind_from_code(code.as_deref()).unwrap_or(ProviderErrorKind::Provider),
        provider,
        status: None,
        code,
        message,
        retry_after: None,
        request_id: None,
        raw: Some(Box::new(raw)),
    }
}
