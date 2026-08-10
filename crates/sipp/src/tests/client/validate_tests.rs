//! Tests the `client::validate` module in `sipp`.
//!
//! Covers deterministic public request validation without endpoint or native
//! execution.

use serde_json::json;

use crate::client::{
    validate, SippEmbedRequest, SippListenRequest, SippQueryRequest, SippSpeakRequest,
    SippTextOptions,
};

#[test]
fn local_requests_reject_extra_fields() {
    let mut request = SippQueryRequest::default();
    request.extra.insert("trace".to_string(), json!(true));
    assert!(matches!(
        validate::local_query(&request),
        Err(crate::client::SippError::InvalidRequest(message))
            if message == "extra fields are not valid for local endpoints"
    ));

    let mut embed = SippEmbedRequest::default();
    embed.extra.insert("normalize".to_string(), json!(true));
    assert!(validate::local_embed(&embed).is_err());
}

#[test]
fn common_text_options_reject_invalid_numbers() {
    assert!(validate::common_text_options(&SippTextOptions {
        max_tokens: Some(0),
        ..Default::default()
    })
    .is_err());
    assert!(validate::common_text_options(&SippTextOptions {
        top_p: Some(1.1),
        ..Default::default()
    })
    .is_err());
}

#[test]
fn speak_rejects_an_explicitly_empty_speaker_reference() {
    assert!(validate::speak(&SippSpeakRequest {
        endpoint: None,
        text: "hello".to_string(),
        language: None,
        speaker_audio: Some(Vec::new()),
        max_duration_ms: None,
    })
    .is_err());
}

#[test]
fn speech_text_and_language_reject_blank_or_padded_values() {
    for language in ["", " ", " en", "en "] {
        assert!(validate::listen(&SippListenRequest {
            endpoint: None,
            audio: vec![1],
            language: Some(language.to_string()),
            max_tokens: None,
        })
        .is_err());
    }

    assert!(validate::speak(&SippSpeakRequest {
        endpoint: None,
        text: " \t".to_string(),
        language: None,
        speaker_audio: None,
        max_duration_ms: None,
    })
    .is_err());

    for language in ["", " ", " en", "en "] {
        assert!(validate::speak(&SippSpeakRequest::new("hello").language(language)).is_err());
    }
    for language in ["en", "english", "nl"] {
        assert!(validate::speak(&SippSpeakRequest::new("hello").language(language)).is_ok());
    }

    assert!(validate::speak(&SippSpeakRequest::new("hello")).is_ok());
}

#[test]
fn speak_rejects_zero_max_duration() {
    assert!(matches!(
        validate::speak(&SippSpeakRequest::new("hello").max_duration_ms(0)),
        Err(crate::client::SippError::InvalidRequest(message))
            if message == "max_duration_ms must be positive"
    ));
}

#[test]
fn listen_rejects_zero_max_tokens() {
    assert!(matches!(
        validate::listen(&SippListenRequest::new([1]).max_tokens(0)),
        Err(crate::client::SippError::InvalidRequest(message))
            if message == "max_tokens must be positive"
    ));
}
