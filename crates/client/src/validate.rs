use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentError, CogentQueryRequest, CogentTextOptions,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/validate_tests.rs"]
mod validate_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) fn common_text_options(options: &CogentTextOptions) -> Result<(), CogentError> {
    if matches!(options.max_tokens, Some(0)) {
        return Err(CogentError::InvalidRequest(
            "max_tokens must be positive".to_string(),
        ));
    }
    finite_optional("temperature", options.temperature)?;
    if options.temperature.is_some_and(|value| value < 0.0) {
        return Err(CogentError::InvalidRequest(
            "temperature must be greater than or equal to zero".to_string(),
        ));
    }
    finite_optional("top_p", options.top_p)?;
    if options
        .top_p
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(CogentError::InvalidRequest(
            "top_p must be between 0 and 1".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn local_query(request: &CogentQueryRequest) -> Result<(), CogentError> {
    common_text_options(&request.options)?;
    reject_gateway_options(&request.gateway_options)
}

pub(crate) fn local_chat(request: &CogentChatRequest) -> Result<(), CogentError> {
    common_text_options(&request.options)?;
    reject_gateway_options(&request.gateway_options)
}

pub(crate) fn local_embed(request: &CogentEmbedRequest) -> Result<(), CogentError> {
    reject_gateway_options(&request.gateway_options)
}

#[cfg(feature = "remote")]
pub(crate) fn remote_query(request: &CogentQueryRequest) -> Result<(), CogentError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(CogentError::InvalidRequest(
            "local text options are not valid for remote endpoints".to_string(),
        ));
    }
    reject_local_only_gateway_options(&request.gateway_options)?;
    Ok(())
}

#[cfg(feature = "remote")]
pub(crate) fn remote_chat(request: &CogentChatRequest) -> Result<(), CogentError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(CogentError::InvalidRequest(
            "local text options are not valid for remote endpoints".to_string(),
        ));
    }
    reject_local_only_gateway_options(&request.gateway_options)?;
    Ok(())
}

#[cfg(feature = "remote")]
pub(crate) fn remote_embed(request: &CogentEmbedRequest) -> Result<(), CogentError> {
    if request.local.has_fields() {
        return Err(CogentError::InvalidRequest(
            "local embed options are not valid for remote endpoints".to_string(),
        ));
    }
    reject_local_only_gateway_options(&request.gateway_options)?;
    Ok(())
}

fn reject_gateway_options(
    gateway_options: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), CogentError> {
    if gateway_options.is_empty() {
        Ok(())
    } else {
        Err(CogentError::InvalidRequest(
            "gateway_options are not valid for local endpoints".to_string(),
        ))
    }
}

#[cfg(feature = "remote")]
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

#[cfg(feature = "remote")]
fn reject_local_only_gateway_options(
    gateway_options: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), CogentError> {
    for key in gateway_options.keys() {
        if LOCAL_ONLY_GATEWAY_FIELDS.contains(&key.as_str()) {
            return Err(CogentError::InvalidRequest(format!(
                "gateway_options cannot contain local-only field: {key}"
            )));
        }
    }
    Ok(())
}

fn finite_optional(name: &'static str, value: Option<f32>) -> Result<(), CogentError> {
    if value.is_some_and(f32::is_finite) || value.is_none() {
        Ok(())
    } else {
        Err(CogentError::InvalidRequest(format!(
            "{name} must be finite"
        )))
    }
}
