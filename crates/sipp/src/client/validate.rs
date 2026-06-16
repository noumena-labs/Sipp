use crate::client::{
    SippChatRequest, SippEmbedRequest, SippError, SippQueryRequest, SippTextOptions,
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
    reject_endpoint_options(&request.endpoint_options, "local endpoints")?;
    reject_provider_options(&request.provider_options, "local endpoints")
}

pub(crate) fn local_chat(request: &SippChatRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    reject_endpoint_options(&request.endpoint_options, "local endpoints")?;
    reject_provider_options(&request.provider_options, "local endpoints")
}

pub(crate) fn local_embed(request: &SippEmbedRequest) -> Result<(), SippError> {
    reject_endpoint_options(&request.endpoint_options, "local endpoints")?;
    reject_provider_options(&request.provider_options, "local endpoints")
}

pub(crate) fn gateway_query(request: &SippQueryRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local text options are not valid for gateway endpoints".to_string(),
        ));
    }
    reject_provider_options(&request.provider_options, "gateway endpoints")?;
    reject_local_only_endpoint_options(&request.endpoint_options)?;
    Ok(())
}

pub(crate) fn gateway_chat(request: &SippChatRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local text options are not valid for gateway endpoints".to_string(),
        ));
    }
    reject_provider_options(&request.provider_options, "gateway endpoints")?;
    reject_local_only_endpoint_options(&request.endpoint_options)?;
    Ok(())
}

pub(crate) fn gateway_embed(request: &SippEmbedRequest) -> Result<(), SippError> {
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local embed options are not valid for gateway endpoints".to_string(),
        ));
    }
    reject_provider_options(&request.provider_options, "gateway endpoints")?;
    reject_local_only_endpoint_options(&request.endpoint_options)?;
    Ok(())
}

#[cfg(feature = "providers")]
pub(crate) fn provider_query(request: &SippQueryRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local text options are not valid for provider endpoints".to_string(),
        ));
    }
    reject_endpoint_options(&request.endpoint_options, "provider endpoints")
}

#[cfg(feature = "providers")]
pub(crate) fn provider_chat(request: &SippChatRequest) -> Result<(), SippError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local text options are not valid for provider endpoints".to_string(),
        ));
    }
    reject_endpoint_options(&request.endpoint_options, "provider endpoints")
}

#[cfg(feature = "providers")]
pub(crate) fn provider_embed(request: &SippEmbedRequest) -> Result<(), SippError> {
    if request.local.has_fields() {
        return Err(SippError::InvalidRequest(
            "local embed options are not valid for provider endpoints".to_string(),
        ));
    }
    reject_endpoint_options(&request.endpoint_options, "provider endpoints")
}

fn reject_endpoint_options(
    endpoint_options: &serde_json::Map<String, serde_json::Value>,
    endpoint_label: &'static str,
) -> Result<(), SippError> {
    if endpoint_options.is_empty() {
        Ok(())
    } else {
        Err(SippError::InvalidRequest(format!(
            "endpoint_options are not valid for {endpoint_label}"
        )))
    }
}

fn reject_provider_options(
    provider_options: &serde_json::Map<String, serde_json::Value>,
    endpoint_label: &'static str,
) -> Result<(), SippError> {
    if provider_options.is_empty() {
        Ok(())
    } else {
        Err(SippError::InvalidRequest(format!(
            "provider_options are not valid for {endpoint_label}"
        )))
    }
}

const LOCAL_ONLY_ENDPOINT_FIELDS: &[&str] = &[
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

fn reject_local_only_endpoint_options(
    endpoint_options: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), SippError> {
    for key in endpoint_options.keys() {
        if LOCAL_ONLY_ENDPOINT_FIELDS.contains(&key.as_str()) {
            return Err(SippError::InvalidRequest(format!(
                "endpoint_options cannot contain local-only field: {key}"
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
