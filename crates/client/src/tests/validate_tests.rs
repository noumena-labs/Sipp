//! Tests the `validate` module in `cogentlm-client`.
//!
//! Covers local/remote request boundary validation and numeric option rejection
//! with synthetic request envelopes instead of endpoint execution.

use serde_json::json;

use super::*;
use crate::{CogentChatRequest, CogentEmbedRequest, CogentQueryRequest, CogentTextOptions};
#[cfg(feature = "providers")]
use crate::{LocalEmbedOptions, LocalTextOptions};

#[test]
fn common_text_options_accept_valid_boundaries() {
    let options = CogentTextOptions {
        max_tokens: Some(1),
        temperature: Some(0.0),
        top_p: Some(1.0),
        ..CogentTextOptions::default()
    };

    common_text_options(&options).expect("valid common text options");
}

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
    let mut query = CogentQueryRequest::default();
    query.remote_options.insert("seed".to_string(), json!(42));

    assert!(matches!(
        local_query(&query),
        Err(CogentError::InvalidRequest(message)) if message.contains("remote_options")
    ));

    let mut chat = CogentChatRequest::default();
    chat.remote_options.insert("seed".to_string(), json!(42));
    assert!(matches!(
        local_chat(&chat),
        Err(CogentError::InvalidRequest(message)) if message.contains("remote_options")
    ));

    let mut embed = CogentEmbedRequest::default();
    embed.remote_options.insert("seed".to_string(), json!(42));
    assert!(matches!(
        local_embed(&embed),
        Err(CogentError::InvalidRequest(message)) if message.contains("remote_options")
    ));
}

#[test]
fn local_requests_accept_empty_remote_options() {
    local_query(&CogentQueryRequest::default()).expect("local query");
    local_chat(&CogentChatRequest::default()).expect("local chat");
    local_embed(&CogentEmbedRequest::default()).expect("local embed");
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
fn remote_text_requests_reject_invalid_common_options() {
    let query = CogentQueryRequest {
        options: CogentTextOptions {
            max_tokens: Some(0),
            ..CogentTextOptions::default()
        },
        ..CogentQueryRequest::default()
    };
    assert!(matches!(
        remote_query(&query),
        Err(CogentError::InvalidRequest(message)) if message.contains("max_tokens")
    ));

    let chat = CogentChatRequest {
        options: CogentTextOptions {
            temperature: Some(f32::INFINITY),
            ..CogentTextOptions::default()
        },
        ..CogentChatRequest::default()
    };
    assert!(matches!(
        remote_chat(&chat),
        Err(CogentError::InvalidRequest(message)) if message.contains("temperature")
    ));
}

#[cfg(feature = "providers")]
#[test]
fn remote_requests_accept_remote_options_without_local_fields() {
    let mut query = CogentQueryRequest::default();
    query.remote_options.insert("seed".to_string(), json!(42));
    remote_query(&query).expect("remote query");

    let mut chat = CogentChatRequest::default();
    chat.remote_options.insert("seed".to_string(), json!(42));
    remote_chat(&chat).expect("remote chat");

    let mut embed = CogentEmbedRequest::default();
    embed.remote_options.insert("seed".to_string(), json!(42));
    remote_embed(&embed).expect("remote embed");
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
