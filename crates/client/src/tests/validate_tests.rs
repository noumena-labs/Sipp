//! Tests the `validate` module in `cogentlm-client`.
//!
//! Covers local/remote request boundary validation and numeric option rejection
//! with synthetic request envelopes instead of endpoint execution.

#[cfg(feature = "remote")]
use cogentlm_engine::engine::SamplingRuntimeConfig;
use serde_json::json;

use super::*;
#[cfg(any(feature = "remote", feature = "providers"))]
use crate::{LocalEmbedOptions, LocalTextOptions};

#[test]
fn local_requests_reject_gateway_options() {
    let mut query = CogentQueryRequest {
        prompt: "hello".to_string(),
        ..CogentQueryRequest::default()
    };
    query
        .gateway_options
        .insert("trace".to_string(), json!("remote-only"));
    assert!(matches!(
        local_query(&query),
        Err(CogentError::InvalidRequest(message))
            if message == "gateway_options are not valid for local endpoints"
    ));

    let mut chat = CogentChatRequest::default();
    chat.gateway_options
        .insert("trace".to_string(), json!("remote-only"));
    assert!(matches!(
        local_chat(&chat),
        Err(CogentError::InvalidRequest(message))
            if message == "gateway_options are not valid for local endpoints"
    ));

    let mut embed = CogentEmbedRequest {
        input: "hello".to_string(),
        ..CogentEmbedRequest::default()
    };
    embed
        .gateway_options
        .insert("trace".to_string(), json!("remote-only"));
    assert!(matches!(
        local_embed(&embed),
        Err(CogentError::InvalidRequest(message))
            if message == "gateway_options are not valid for local endpoints"
    ));
}

#[test]
fn local_requests_reject_provider_options() {
    let mut query = CogentQueryRequest {
        prompt: "hello".to_string(),
        ..CogentQueryRequest::default()
    };
    query.provider_options.insert("seed".to_string(), json!(1));
    assert!(matches!(
        local_query(&query),
        Err(CogentError::InvalidRequest(message))
            if message == "provider_options are not valid for local endpoints"
    ));

    let mut embed = CogentEmbedRequest {
        input: "hello".to_string(),
        ..CogentEmbedRequest::default()
    };
    embed
        .provider_options
        .insert("input_type".to_string(), json!("query"));
    assert!(matches!(
        local_embed(&embed),
        Err(CogentError::InvalidRequest(message))
            if message == "provider_options are not valid for local endpoints"
    ));
}

#[test]
fn text_options_reject_out_of_range_sampling_values() {
    let mut options = CogentTextOptions {
        temperature: Some(-0.1),
        ..CogentTextOptions::default()
    };
    assert!(matches!(
        common_text_options(&options),
        Err(CogentError::InvalidRequest(message))
            if message == "temperature must be greater than or equal to zero"
    ));

    options.temperature = None;
    options.top_p = Some(1.1);
    assert!(matches!(
        common_text_options(&options),
        Err(CogentError::InvalidRequest(message))
            if message == "top_p must be between 0 and 1"
    ));
}

#[cfg(feature = "remote")]
#[test]
fn remote_text_requests_reject_local_only_fields() {
    let request = CogentQueryRequest {
        prompt: "hello".to_string(),
        local: LocalTextOptions {
            context_key: Some("ctx".to_string()),
            grammar: Some("root ::= \"ok\"".to_string()),
            json_schema: Some("{}".to_string()),
            sampling: Some(SamplingRuntimeConfig::default()),
            media: vec![vec![1, 2, 3]],
        },
        ..CogentQueryRequest::default()
    };
    assert!(matches!(
        remote_query(&request),
        Err(CogentError::InvalidRequest(message))
            if message == "local text options are not valid for remote endpoints"
    ));

    let request = CogentChatRequest {
        local: LocalTextOptions {
            context_key: Some("ctx".to_string()),
            ..LocalTextOptions::default()
        },
        ..CogentChatRequest::default()
    };
    assert!(matches!(
        remote_chat(&request),
        Err(CogentError::InvalidRequest(message))
            if message == "local text options are not valid for remote endpoints"
    ));
}

#[cfg(feature = "remote")]
#[test]
fn remote_embed_requests_reject_local_only_fields() {
    let request = CogentEmbedRequest {
        input: "hello".to_string(),
        local: LocalEmbedOptions {
            context_key: Some("ctx".to_string()),
            normalize: Some(true),
        },
        ..CogentEmbedRequest::default()
    };

    assert!(matches!(
        remote_embed(&request),
        Err(CogentError::InvalidRequest(message))
            if message == "local embed options are not valid for remote endpoints"
    ));
}

#[cfg(feature = "remote")]
#[test]
fn remote_requests_reject_local_only_gateway_options() {
    let mut query = CogentQueryRequest {
        prompt: "hello".to_string(),
        ..CogentQueryRequest::default()
    };
    query
        .gateway_options
        .insert("grammar".to_string(), json!("root ::= \"ok\""));
    assert!(matches!(
        remote_query(&query),
        Err(CogentError::InvalidRequest(message))
            if message == "gateway_options cannot contain local-only field: grammar"
    ));

    let mut embed = CogentEmbedRequest {
        input: "hello".to_string(),
        ..CogentEmbedRequest::default()
    };
    embed
        .gateway_options
        .insert("normalize".to_string(), json!(true));
    assert!(matches!(
        remote_embed(&embed),
        Err(CogentError::InvalidRequest(message))
            if message == "gateway_options cannot contain local-only field: normalize"
    ));
}

#[cfg(feature = "remote")]
#[test]
fn remote_requests_reject_provider_options() {
    let mut request = CogentQueryRequest {
        prompt: "hello".to_string(),
        ..CogentQueryRequest::default()
    };
    request
        .provider_options
        .insert("seed".to_string(), json!(1));

    assert!(matches!(
        remote_query(&request),
        Err(CogentError::InvalidRequest(message))
            if message == "provider_options are not valid for remote endpoints"
    ));
}

#[cfg(feature = "providers")]
#[test]
fn provider_requests_reject_local_and_gateway_options() {
    let request = CogentQueryRequest {
        prompt: "hello".to_string(),
        local: LocalTextOptions {
            context_key: Some("ctx".to_string()),
            ..LocalTextOptions::default()
        },
        ..CogentQueryRequest::default()
    };
    assert!(matches!(
        provider_query(&request),
        Err(CogentError::InvalidRequest(message))
            if message == "local text options are not valid for provider endpoints"
    ));

    let mut request = CogentEmbedRequest {
        input: "hello".to_string(),
        ..CogentEmbedRequest::default()
    };
    request
        .gateway_options
        .insert("trace".to_string(), json!("gateway-only"));
    assert!(matches!(
        provider_embed(&request),
        Err(CogentError::InvalidRequest(message))
            if message == "gateway_options are not valid for provider endpoints"
    ));
}
