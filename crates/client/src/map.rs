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
        .stream_tokens(request.stream_tokens))
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
pub(crate) fn provider_text_response(
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
pub(crate) fn provider_embedding_response(
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
mod tests {
    use cogentlm_engine::engine::SamplingRuntimeConfig;

    use super::*;
    use crate::{CogentQueryRequest, LocalTextOptions};

    #[test]
    fn common_sampling_without_local_sampler_builds_sparse_patch() {
        let request = CogentQueryRequest {
            prompt: "hello".to_string(),
            options: CogentTextOptions {
                temperature: Some(0.2),
                ..CogentTextOptions::default()
            },
            ..CogentQueryRequest::default()
        };

        let request = local_query_request(request).expect("local query request");

        let Some(RequestSampling::Patch(patch)) = request.options.sampling else {
            panic!("common-only sampling should use sparse patch");
        };
        assert_eq!(patch.temperature, Some(0.2));
        assert_eq!(patch.top_p, None);
    }

    #[test]
    fn common_sampling_conflicts_with_different_explicit_local_sampler() {
        let request = CogentQueryRequest {
            prompt: "hello".to_string(),
            options: CogentTextOptions {
                temperature: Some(0.2),
                ..CogentTextOptions::default()
            },
            local: LocalTextOptions {
                sampling: Some(SamplingRuntimeConfig {
                    temperature: Some(0.7),
                    ..SamplingRuntimeConfig::default()
                }),
                ..LocalTextOptions::default()
            },
            ..CogentQueryRequest::default()
        };

        let error = match local_query_request(request) {
            Err(error) => error,
            Ok(_) => panic!("conflict must reject"),
        };

        assert!(
            matches!(error, CogentError::InvalidRequest(message) if message.contains("temperature"))
        );
    }

    #[test]
    fn usage_from_stats_omits_negative_counts_and_unchecked_total() {
        let usage = usage_from_stats(RequestStats {
            input_tokens: -1,
            output_tokens: 3,
            ..RequestStats::default()
        });

        assert_eq!(usage.input_tokens, None);
        assert_eq!(usage.output_tokens, Some(3));
        assert_eq!(usage.total_tokens, None);
    }
}
