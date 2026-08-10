use crate::client::{
    SippChatRequest, SippEmbedRequest, SippError, SippListenRequest, SippQueryRequest,
    SippSpeakRequest, SippTextOptions,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/client/validate_tests.rs"]
mod validate_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) fn common_text_options(options: &SippTextOptions) -> Result<(), SippError> {
    if matches!(options.max_tokens, Some(0)) {
        return Err(SippError::InvalidRequest(
            "max_tokens must be positive".to_string(),
        ));
    }
    finite_optional("temperature", options.temperature)?;
    if options.temperature.is_some_and(|value| value < 0.0) {
        return Err(SippError::InvalidRequest(
            "temperature must be greater than or equal to zero".to_string(),
        ));
    }
    finite_optional("top_p", options.top_p)?;
    if options
        .top_p
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(SippError::InvalidRequest(
            "top_p must be between 0 and 1".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn local_query(request: &SippQueryRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    reject_extra(&request.extra, "local endpoints")
}

pub(crate) fn local_chat(request: &SippChatRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    reject_extra(&request.extra, "local endpoints")
}

pub(crate) fn local_embed(request: &SippEmbedRequest) -> Result<(), SippError> {
    reject_extra(&request.extra, "local endpoints")
}

pub(crate) fn listen(request: &SippListenRequest) -> Result<(), SippError> {
    if request.audio.is_empty() {
        return Err(SippError::InvalidRequest(
            "listen audio must not be empty".to_string(),
        ));
    }
    if matches!(request.max_tokens, Some(0)) {
        return Err(SippError::InvalidRequest(
            "max_tokens must be positive".to_string(),
        ));
    }
    if let Some(language) = &request.language {
        validate_language("listen", language)?;
    }
    Ok(())
}

pub(crate) fn speak(request: &SippSpeakRequest) -> Result<(), SippError> {
    if request.text.trim().is_empty() {
        return Err(SippError::InvalidRequest(
            "speak text must not be empty".to_string(),
        ));
    }
    if let Some(language) = &request.language {
        validate_language("speak", language)?;
    }
    if request.speaker_audio.as_ref().is_some_and(Vec::is_empty) {
        return Err(SippError::InvalidRequest(
            "speak speaker audio must not be empty".to_string(),
        ));
    }
    if matches!(request.max_duration_ms, Some(0)) {
        return Err(SippError::InvalidRequest(
            "max_duration_ms must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_language(operation: &'static str, language: &str) -> Result<(), SippError> {
    if language.trim().is_empty() {
        return Err(SippError::InvalidRequest(format!(
            "{operation} language must not be empty"
        )));
    }
    if language.trim() != language {
        return Err(SippError::InvalidRequest(format!(
            "{operation} language must not contain surrounding whitespace"
        )));
    }
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn gateway_query(request: &SippQueryRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local text options are not valid for gateway endpoints".to_string(),
        ));
    }
    reject_local_only_extra(&request.extra)?;
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn gateway_chat(request: &SippChatRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local text options are not valid for gateway endpoints".to_string(),
        ));
    }
    reject_local_only_extra(&request.extra)?;
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn gateway_embed(request: &SippEmbedRequest) -> Result<(), SippError> {
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local embed options are not valid for gateway endpoints".to_string(),
        ));
    }
    reject_local_only_extra(&request.extra)?;
    Ok(())
}

#[cfg(all(feature = "providers", not(target_family = "wasm")))]
pub(crate) fn provider_query(request: &SippQueryRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local text options are not valid for provider endpoints".to_string(),
        ));
    }
    Ok(())
}

#[cfg(all(feature = "providers", not(target_family = "wasm")))]
pub(crate) fn provider_chat(request: &SippChatRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local text options are not valid for provider endpoints".to_string(),
        ));
    }
    Ok(())
}

#[cfg(all(feature = "providers", not(target_family = "wasm")))]
pub(crate) fn provider_embed(request: &SippEmbedRequest) -> Result<(), SippError> {
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local embed options are not valid for provider endpoints".to_string(),
        ));
    }
    Ok(())
}

fn reject_extra(
    extra: &serde_json::Map<String, serde_json::Value>,
    endpoint_label: &'static str,
) -> Result<(), SippError> {
    if extra.is_empty() {
        Ok(())
    } else {
        Err(SippError::InvalidRequest(format!(
            "extra fields are not valid for {endpoint_label}"
        )))
    }
}

#[cfg(not(target_family = "wasm"))]
const LOCAL_ONLY_EXTRA_FIELDS: &[&str] = &[
    "context_key",
    "contextKey",
    "grammar",
    "json_schema",
    "jsonSchema",
    "sampling",
    "media",
    "normalize",
    "local",
];

#[cfg(not(target_family = "wasm"))]
fn reject_local_only_extra(
    extra: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), SippError> {
    for key in extra.keys() {
        if LOCAL_ONLY_EXTRA_FIELDS.contains(&key.as_str()) {
            return Err(SippError::InvalidRequest(format!(
                "extra cannot contain local-only field: {key}"
            )));
        }
    }
    Ok(())
}

fn finite_optional(name: &'static str, value: Option<f32>) -> Result<(), SippError> {
    if value.is_some_and(f32::is_finite) || value.is_none() {
        Ok(())
    } else {
        Err(SippError::InvalidRequest(format!("{name} must be finite")))
    }
}
