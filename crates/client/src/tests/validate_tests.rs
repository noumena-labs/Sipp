//! Tests the `validate` module in `cogentlm-client`.
//!
//! Covers endpoint resolution, remote configuration, facade validation, and run wrappers with deterministic fakes rather than a live local engine.

use serde_json::json;

use super::*;
use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentQueryRequest, CogentTextOptions,
    LocalEmbedOptions, LocalTextOptions,
};

#[test]
fn common_text_options_reject_zero_and_non_finite_values() {
    let zero_tokens = CogentTextOptions {
        max_tokens: Some(0),
        ..CogentTextOptions::default()
    };
    assert!(matches!(
        common_text_options(&zero_tokens),
        Err(CogentError::InvalidRequest(message)) if message.contains("max_tokens")
    ));

    let bad_temperature = CogentTextOptions {
        temperature: Some(f32::NAN),
        ..CogentTextOptions::default()
    };
    assert!(matches!(
        common_text_options(&bad_temperature),
        Err(CogentError::InvalidRequest(message)) if message.contains("temperature")
    ));

    let bad_top_p = CogentTextOptions {
        top_p: Some(f32::INFINITY),
        ..CogentTextOptions::default()
    };
    assert!(matches!(
        common_text_options(&bad_top_p),
        Err(CogentError::InvalidRequest(message)) if message.contains("top_p")
    ));
}

#[test]
fn local_requests_reject_remote_options() {
    let mut request = CogentQueryRequest::default();
    request.remote_options.insert("seed".to_string(), json!(42));

    assert!(matches!(
        local_query(&request),
        Err(CogentError::InvalidRequest(message)) if message.contains("remote_options")
    ));
}

#[cfg(feature = "providers")]
#[test]
fn remote_text_requests_reject_local_options() {
    let query = CogentQueryRequest {
        local: LocalTextOptions {
            context_key: Some("ctx".to_string()),
            ..LocalTextOptions::default()
        },
        ..CogentQueryRequest::default()
    };
    assert!(matches!(
        remote_query(&query),
        Err(CogentError::InvalidRequest(message)) if message.contains("local text options")
    ));

    let chat = CogentChatRequest {
        local: LocalTextOptions {
            grammar: Some("root ::= \"ok\"".to_string()),
            ..LocalTextOptions::default()
        },
        ..CogentChatRequest::default()
    };
    assert!(matches!(
        remote_chat(&chat),
        Err(CogentError::InvalidRequest(message)) if message.contains("local text options")
    ));
}

#[cfg(feature = "providers")]
#[test]
fn remote_embed_requests_reject_local_options() {
    let request = CogentEmbedRequest {
        local: LocalEmbedOptions {
            normalize: Some(true),
            ..LocalEmbedOptions::default()
        },
        ..CogentEmbedRequest::default()
    };

    assert!(matches!(
        remote_embed(&request),
        Err(CogentError::InvalidRequest(message)) if message.contains("local embed options")
    ));
}
