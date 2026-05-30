use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentError, CogentQueryRequest, CogentTextOptions,
};

pub(crate) fn common_text_options(options: &CogentTextOptions) -> Result<(), CogentError> {
    if matches!(options.max_tokens, Some(0)) {
        return Err(CogentError::InvalidRequest(
            "max_tokens must be positive".to_string(),
        ));
    }
    finite_optional("temperature", options.temperature)?;
    finite_optional("top_p", options.top_p)
}

pub(crate) fn local_query(request: &CogentQueryRequest) -> Result<(), CogentError> {
    common_text_options(&request.options)?;
    reject_provider_options(&request.provider_options)
}

pub(crate) fn local_chat(request: &CogentChatRequest) -> Result<(), CogentError> {
    common_text_options(&request.options)?;
    reject_provider_options(&request.provider_options)
}

pub(crate) fn local_embed(request: &CogentEmbedRequest) -> Result<(), CogentError> {
    reject_provider_options(&request.provider_options)
}

#[cfg(feature = "providers")]
pub(crate) fn provider_query(request: &CogentQueryRequest) -> Result<(), CogentError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(CogentError::InvalidRequest(
            "local text options are not valid for provider endpoints".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "providers")]
pub(crate) fn provider_chat(request: &CogentChatRequest) -> Result<(), CogentError> {
    common_text_options(&request.options)?;
    if request.local.has_fields() {
        return Err(CogentError::InvalidRequest(
            "local text options are not valid for provider endpoints".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "providers")]
pub(crate) fn provider_embed(request: &CogentEmbedRequest) -> Result<(), CogentError> {
    if request.local.has_fields() {
        return Err(CogentError::InvalidRequest(
            "local embed options are not valid for provider endpoints".to_string(),
        ));
    }
    Ok(())
}

fn reject_provider_options(
    provider_options: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), CogentError> {
    if provider_options.is_empty() {
        Ok(())
    } else {
        Err(CogentError::InvalidRequest(
            "provider_options are not valid for local endpoints".to_string(),
        ))
    }
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
