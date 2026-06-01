use cogentlm_core::TokenUsage;
use cogentlm_engine::engine::{
    EmbedOptions, EmbedRequest, EmbeddingResult, GenerationResult, QueryOptions, QueryRequest,
    RequestSampling, RequestStats, SamplingRuntimePatch, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};

use crate::{
    CogentEmbeddingResponse, CogentError, CogentQueryRequest, CogentTextOptions,
    CogentTextResponse, EndpointRef, LocalEmbedOptions, LocalTextOptions,
};

pub(crate) fn local_query_request(
    request: CogentQueryRequest,
) -> Result<QueryRequest, CogentError> {
    let options = local_query_options(request.options, request.local)?;
    Ok(QueryRequest::new(request.prompt)
        .options(options)
        .emit_tokens(request.emit_tokens))
}

pub(crate) fn local_chat_options(
    options: CogentTextOptions,
    local: LocalTextOptions,
) -> Result<QueryOptions, CogentError> {
    local_query_options(options, local)
}

pub(crate) fn local_embed_request(input: String, local: LocalEmbedOptions) -> EmbedRequest {
    EmbedRequest {
        input,
        options: EmbedOptions {
            normalize: local.normalize.unwrap_or(true),
            context_key: local.context_key,
        },
    }
}

pub(crate) fn text_response(endpoint: EndpointRef, result: GenerationResult) -> CogentTextResponse {
    CogentTextResponse {
        endpoint,
        text: result.text,
        finish_reason: result.finish_reason,
        usage: Some(usage_from_stats(result.stats)),
        local_stats: Some(result.stats),
    }
}

pub(crate) fn embedding_response(
    endpoint: EndpointRef,
    result: EmbeddingResult,
) -> CogentEmbeddingResponse {
    CogentEmbeddingResponse {
        endpoint,
        values: result.values,
        usage: Some(usage_from_stats(result.stats)),
        local_stats: Some(result.stats),
        pooling: Some(result.pooling),
        normalized: Some(result.normalized),
    }
}

#[cfg(feature = "providers")]
pub(crate) fn remote_text_response(
    endpoint: EndpointRef,
    response: cogentlm_providers::ProviderResponse<cogentlm_providers::ProviderTextOutput>,
) -> CogentTextResponse {
    CogentTextResponse {
        endpoint,
        text: response.result.text,
        finish_reason: response.result.finish_reason,
        usage: response.usage,
        local_stats: None,
    }
}

#[cfg(feature = "providers")]
pub(crate) fn remote_embedding_response(
    endpoint: EndpointRef,
    response: cogentlm_providers::ProviderResponse<cogentlm_providers::ProviderEmbeddingOutput>,
) -> CogentEmbeddingResponse {
    CogentEmbeddingResponse {
        endpoint,
        values: response.result.values,
        usage: response.usage,
        local_stats: None,
        pooling: None,
        normalized: None,
    }
}

pub(crate) fn usage_from_stats(stats: RequestStats) -> TokenUsage {
    let input_tokens = nonnegative_i32_to_u32(stats.input_tokens);
    let output_tokens = nonnegative_i32_to_u32(stats.output_tokens);
    let total_tokens = match (input_tokens, output_tokens) {
        (Some(input), Some(output)) => input.checked_add(output),
        _ => None,
    };
    TokenUsage {
        input_tokens,
        output_tokens,
        total_tokens,
    }
}

fn local_query_options(
    options: CogentTextOptions,
    local: LocalTextOptions,
) -> Result<QueryOptions, CogentError> {
    let max_tokens = match options.max_tokens {
        Some(max_tokens) => i32::try_from(max_tokens).map_err(|_| {
            CogentError::InvalidRequest("local max_tokens exceeds i32::MAX".to_string())
        })?,
        None => DEFAULT_MAX_TOKENS,
    };
    let sampling = local_sampling(options.temperature, options.top_p, local.sampling)?;

    Ok(QueryOptions {
        context_key: local
            .context_key
            .unwrap_or_else(|| DEFAULT_CONTEXT_KEY.to_string()),
        max_tokens,
        grammar: local.grammar.unwrap_or_default(),
        json_schema: local.json_schema.unwrap_or_default(),
        stop: options.stop,
        sampling,
        media: local.media,
    })
}

fn local_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    sampling: Option<cogentlm_engine::engine::SamplingRuntimeConfig>,
) -> Result<Option<RequestSampling>, CogentError> {
    if let Some(mut sampling) = sampling {
        merge_sampling_field("temperature", &mut sampling.temperature, temperature)?;
        merge_sampling_field("top_p", &mut sampling.top_p, top_p)?;
        return Ok(Some(RequestSampling::Full(sampling)));
    }

    if temperature.is_some() || top_p.is_some() {
        Ok(Some(RequestSampling::Patch(SamplingRuntimePatch {
            temperature,
            top_p,
        })))
    } else {
        Ok(None)
    }
}

fn merge_sampling_field(
    name: &'static str,
    target: &mut Option<f32>,
    value: Option<f32>,
) -> Result<(), CogentError> {
    let Some(value) = value else {
        return Ok(());
    };
    match target {
        Some(existing) if *existing != value => Err(CogentError::InvalidRequest(format!(
            "common {name} conflicts with local sampling.{name}"
        ))),
        Some(_) => Ok(()),
        None => {
            *target = Some(value);
            Ok(())
        }
    }
}

fn nonnegative_i32_to_u32(value: i32) -> Option<u32> {
    u32::try_from(value).ok()
}

#[cfg(test)]
#[path = "tests/map_tests.rs"]
mod map_tests;
