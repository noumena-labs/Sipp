//! Tests the `engine::driver::request` module in `sipp`.
//!
//! Covers driver futures, command handling, event emission, and request mapping with model-free channels or explicitly ignored model smoke tests.

use std::sync::{Arc, Mutex};

use super::{render_messages_json, start_chat, start_embed, ChatMessage, ChatRequest, ChatRole};
use crate::engine::protocol::{EmbedOptions, EmbedRequest, ModelClass};
use crate::error::Error;
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;

#[test]
fn render_messages_json_preserves_role_and_content_order() {
    let messages = [
        ChatMessage::new(ChatRole::System, "policy"),
        ChatMessage::new(ChatRole::User, "hello"),
        ChatMessage::new(ChatRole::Assistant, "hi"),
    ];

    let json = render_messages_json(&messages).expect("messages json");

    assert_eq!(
        json,
        r#"[{"content":"policy","role":"system"},{"content":"hello","role":"user"},{"content":"hi","role":"assistant"}]"#
    );
}

#[test]
fn chat_rejects_models_without_chat_template() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let subscribers = Arc::new(Mutex::new(Vec::new()));

    let request = ChatRequest::new(vec![ChatMessage::new(ChatRole::User, "hello")]);
    let error = match start_chat(&mut runtime, request, None, &subscribers) {
        Err(error) => error,
        Ok(_) => panic!("chat() must reject when has_chat_template is false"),
    };

    assert!(
        matches!(&error, Error::UnsupportedOperation { operation: "chat", reason }
            if reason.contains("no chat template")),
        "expected chat() to reject with UnsupportedOperation; got: {error:?}"
    );
}

#[test]
fn embed_rejects_decoder_only_without_embedding_context() {
    // test_runtime defaults to DecoderOnly + embedding_context=false: that's
    // the standard text-generation case, so embed() must refuse before
    // tokenization.
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let subscribers = Arc::new(Mutex::new(Vec::new()));

    let request = EmbedRequest {
        input: "hello".to_string(),
        options: EmbedOptions::default(),
    };
    let error = match start_embed(&mut runtime, request, &subscribers) {
        Err(error) => error,
        Ok(_) => panic!("embed() must reject when embeddings=true is not set"),
    };

    assert!(
        matches!(&error, Error::UnsupportedOperation { operation: "embed", reason }
            if reason.contains("embeddings=true")),
        "expected embed() to reject; got: {error:?}"
    );
}

#[test]
fn embed_rejects_encoder_decoder_models() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.capabilities.class = ModelClass::EncoderDecoder;
    runtime.capabilities.decoder_start_token = Some(0);
    let subscribers = Arc::new(Mutex::new(Vec::new()));

    let request = EmbedRequest {
        input: "hello".to_string(),
        options: EmbedOptions::default(),
    };
    let error = match start_embed(&mut runtime, request, &subscribers) {
        Err(error) => error,
        Ok(_) => panic!("embed() must reject EncoderDecoder models"),
    };

    assert!(
        matches!(&error, Error::UnsupportedOperation { operation: "embed", reason }
            if reason.contains("encoder-decoder")),
        "expected embed() to reject; got: {error:?}"
    );
}
