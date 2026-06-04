//! Tests the `map` module in `cogentlm-client`.
//!
//! Covers local request/response translation, sampling merges, and token usage
//! derivation with synthetic engine protocol values instead of native execution.

use cogentlm_core::FinishReason;
use cogentlm_engine::engine::{
    EmbeddingResult, GenerationResult, PoolingType, SamplingRuntimeConfig, DEFAULT_CONTEXT_KEY,
    DEFAULT_MAX_TOKENS,
};
#[cfg(feature = "remote")]
use cogentlm_remote::{
    GatewayEmbeddingOutput, GatewayResponse, GatewayResponseMetadata, GatewayTextOutput, TokenUsage,
};
#[cfg(feature = "remote")]
use serde_json::json;

use super::*;
use crate::{CogentQueryRequest, LocalEmbedOptions, LocalTextOptions};

#[test]
fn local_query_request_maps_defaults_and_explicit_fields() {
    let request = local_query_request(CogentQueryRequest {
        prompt: "hello".to_string(),
        options: CogentTextOptions {
            max_tokens: Some(12),
            temperature: Some(0.4),
            top_p: Some(0.8),
            stop: vec!["stop".to_string()],
        },
        local: LocalTextOptions {
            context_key: Some("ctx".to_string()),
            grammar: Some("root ::= \"ok\"".to_string()),
            json_schema: Some("{\"type\":\"object\"}".to_string()),
            media: vec![vec![1, 2, 3]],
            ..LocalTextOptions::default()
        },
        emit_tokens: true,
        ..CogentQueryRequest::default()
    })
    .expect("local query request");

    assert_eq!(request.prompt, "hello");
    assert_eq!(request.options.context_key, "ctx");
    assert_eq!(request.options.max_tokens, 12);
    assert_eq!(request.options.grammar, "root ::= \"ok\"");
    assert_eq!(request.options.json_schema, "{\"type\":\"object\"}");
    assert_eq!(request.options.stop, vec!["stop"]);
    assert_eq!(request.options.media, vec![vec![1, 2, 3]]);
    assert!(request.emit_tokens);

    let Some(RequestSampling::Patch(patch)) = request.options.sampling else {
        panic!("common sampling should create patch");
    };
    assert_eq!(patch.temperature, Some(0.4));
    assert_eq!(patch.top_p, Some(0.8));
}

#[test]
fn local_query_request_uses_default_options_without_sampling() {
    let request = local_query_request(CogentQueryRequest {
        prompt: "hello".to_string(),
        ..CogentQueryRequest::default()
    })
    .expect("local query request");

    assert_eq!(request.options.context_key, DEFAULT_CONTEXT_KEY);
    assert_eq!(request.options.max_tokens, DEFAULT_MAX_TOKENS);
    assert!(request.options.grammar.is_empty());
    assert!(request.options.json_schema.is_empty());
    assert!(request.options.stop.is_empty());
    assert!(request.options.sampling.is_none());
    assert!(request.options.media.is_empty());
    assert!(!request.emit_tokens);
}

#[test]
fn local_max_tokens_must_fit_engine_i32_limit() {
    let error = match local_query_request(CogentQueryRequest {
        prompt: "hello".to_string(),
        options: CogentTextOptions {
            max_tokens: Some(i32::MAX as u32 + 1),
            ..CogentTextOptions::default()
        },
        ..CogentQueryRequest::default()
    }) {
        Ok(_) => panic!("too-large max_tokens must reject"),
        Err(error) => error,
    };

    assert!(
        matches!(error, CogentError::InvalidRequest(message) if message.contains("max_tokens"))
    );
}

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
fn common_sampling_merges_into_local_sampler_when_fields_match_or_are_missing() {
    let request = CogentQueryRequest {
        prompt: "hello".to_string(),
        options: CogentTextOptions {
            temperature: Some(0.2),
            top_p: Some(0.9),
            ..CogentTextOptions::default()
        },
        local: LocalTextOptions {
            sampling: Some(SamplingRuntimeConfig {
                temperature: Some(0.2),
                top_p: None,
                ..SamplingRuntimeConfig::default()
            }),
            ..LocalTextOptions::default()
        },
        ..CogentQueryRequest::default()
    };

    let request = local_query_request(request).expect("local query request");

    let Some(RequestSampling::Full(sampling)) = request.options.sampling else {
        panic!("local sampling should remain a full config");
    };
    assert_eq!(sampling.temperature, Some(0.2));
    assert_eq!(sampling.top_p, Some(0.9));
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
fn local_chat_options_reuse_query_option_mapping() {
    let options = local_chat_options(
        CogentTextOptions {
            max_tokens: Some(4),
            stop: vec!["done".to_string()],
            ..CogentTextOptions::default()
        },
        LocalTextOptions {
            context_key: Some("chat-ctx".to_string()),
            ..LocalTextOptions::default()
        },
    )
    .expect("local chat options");

    assert_eq!(options.context_key, "chat-ctx");
    assert_eq!(options.max_tokens, 4);
    assert_eq!(options.stop, vec!["done"]);
}

#[test]
fn local_embed_request_maps_defaults_and_overrides() {
    let default_request = local_embed_request("hello".to_string(), LocalEmbedOptions::default());
    assert_eq!(default_request.input, "hello");
    assert!(default_request.options.normalize);
    assert!(default_request.options.context_key.is_none());

    let explicit_request = local_embed_request(
        "world".to_string(),
        LocalEmbedOptions {
            context_key: Some("embed-ctx".to_string()),
            normalize: Some(false),
        },
    );
    assert_eq!(explicit_request.input, "world");
    assert!(!explicit_request.options.normalize);
    assert_eq!(
        explicit_request.options.context_key.as_deref(),
        Some("embed-ctx")
    );
}

#[test]
fn local_responses_preserve_stats_and_endpoint_metadata() {
    let endpoint = EndpointRef::Local {
        id: "local".to_string(),
    };
    let stats = RequestStats {
        input_tokens: 2,
        output_tokens: 3,
        ..RequestStats::default()
    };

    let text = text_response(
        endpoint.clone(),
        GenerationResult {
            id: "text".to_string(),
            text: "done".to_string(),
            finish_reason: FinishReason::Length,
            stats,
        },
    );
    assert_eq!(text.endpoint, endpoint);
    assert_eq!(text.text, "done");
    assert_eq!(text.finish_reason, FinishReason::Length);
    assert_eq!(text.usage.expect("usage").total_tokens, Some(5));
    assert_eq!(text.local_stats, Some(stats));

    let endpoint = EndpointRef::Local {
        id: "embed".to_string(),
    };
    let embedding = embedding_response(
        endpoint.clone(),
        EmbeddingResult {
            id: "embed".to_string(),
            values: vec![1.0, 2.0],
            pooling: PoolingType::Mean,
            normalized: true,
            stats,
        },
    );
    assert_eq!(embedding.endpoint, endpoint);
    assert_eq!(embedding.values, vec![1.0, 2.0]);
    assert_eq!(embedding.usage.expect("usage").total_tokens, Some(5));
    assert_eq!(embedding.local_stats, Some(stats));
    assert_eq!(embedding.pooling, Some(PoolingType::Mean));
    assert_eq!(embedding.normalized, Some(true));
}

#[cfg(feature = "remote")]
#[test]
fn remote_responses_drop_local_metadata_and_preserve_gateway_usage() {
    let metadata = GatewayResponseMetadata {
        model: "remote-model".to_string(),
        request_id: Some("req".to_string()),
        response_id: Some("resp".to_string()),
        finish_reason_raw: None,
        raw: json!({}),
    };
    let usage = Some(TokenUsage {
        input_tokens: Some(1),
        output_tokens: Some(2),
        total_tokens: Some(3),
    });
    let endpoint = EndpointRef::Remote {
        id: "remote".to_string(),
    };

    let text = remote_text_response(
        endpoint.clone(),
        GatewayResponse {
            result: GatewayTextOutput {
                text: "done".to_string(),
                finish_reason: FinishReason::Stop,
            },
            usage,
            metadata: metadata.clone(),
        },
    );
    assert_eq!(text.endpoint, endpoint);
    assert_eq!(text.text, "done");
    assert_eq!(text.usage, usage);
    assert!(text.local_stats.is_none());

    let endpoint = EndpointRef::Remote {
        id: "embed".to_string(),
    };
    let embedding = remote_embedding_response(
        endpoint.clone(),
        GatewayResponse {
            result: GatewayEmbeddingOutput {
                values: vec![1.0, 2.0],
            },
            usage,
            metadata,
        },
    );
    assert_eq!(embedding.endpoint, endpoint);
    assert_eq!(embedding.values, vec![1.0, 2.0]);
    assert_eq!(embedding.usage, usage);
    assert!(embedding.local_stats.is_none());
    assert!(embedding.pooling.is_none());
    assert!(embedding.normalized.is_none());
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
