//! Unit tests for the parent module.

use super::super::*;
use crate::engine::{
    GenerateOptions, RequestSampling, SamplingRuntimeConfig, DEFAULT_CONTEXT_KEY,
    DEFAULT_MAX_TOKENS,
};
use futures::executor::block_on;

#[test]
fn query_options_default_matches_public_completion_defaults() {
    let options = QueryOptions::default();

    assert_eq!(options.context_key, DEFAULT_CONTEXT_KEY);
    assert_eq!(options.max_tokens, DEFAULT_MAX_TOKENS);
    assert!(options.grammar.is_empty());
    assert!(options.json_schema.is_empty());
    assert!(options.stop.is_empty());
    assert!(options.sampling.is_none());
    assert!(options.media.is_empty());
}

#[test]
fn generate_options_convert_to_query_options() {
    let options = QueryOptions::from(GenerateOptions {
        max_tokens: 7,
        stream: true,
        stop: vec!["END".to_string()],
        sampling: Some(SamplingRuntimeConfig {
            temperature: Some(0.1),
            ..SamplingRuntimeConfig::default()
        }),
        grammar: Some("root ::= \"x\"".to_string()),
        json_schema: Some("{}".to_string()),
        cache_key: Some("ctx".to_string()),
    });

    assert_eq!(options.context_key, "ctx");
    assert_eq!(options.max_tokens, 7);
    assert_eq!(options.grammar, "root ::= \"x\"");
    assert_eq!(options.json_schema, "{}");
    assert_eq!(options.stop, vec!["END"]);
    let Some(RequestSampling::Full(sampling)) = &options.sampling else {
        panic!("generate options should map to a full sampling override");
    };
    assert_eq!(sampling.temperature, Some(0.1));
}

#[test]
fn generate_options_without_cache_key_uses_default_context() {
    let options = QueryOptions::from(GenerateOptions {
        cache_key: None,
        ..GenerateOptions::default()
    });

    assert_eq!(options.context_key, DEFAULT_CONTEXT_KEY);
}

#[test]
fn query_request_defaults_options() {
    let request = QueryRequest::new("hello");

    assert_eq!(request.prompt, "hello");
    assert_eq!(request.options, QueryOptions::default());
}

#[test]
fn engine_handle_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<CogentEngine>();
}

/// End-to-end Phase 3 verification: load the repo-root T5 fixture, confirm
/// classification, run an actual encoder-decoder query, and confirm chat()
/// rejects with `UnsupportedOperation`. Ignored by default because it loads a
/// real model and runs llama.cpp under the test harness — run with
/// `cargo test -p cogentlm-engine -- --ignored t5_encoder_decoder_end_to_end`.
#[test]
#[ignore]
fn t5_encoder_decoder_end_to_end() {
    use std::path::PathBuf;

    use crate::engine::protocol::ModelClass;
    use crate::engine::{ChatMessage, ChatRequest, ChatRole, NativeRuntimeConfig};
    use crate::error::Error;

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("..")
        .join("t5-small-f16.gguf");
    assert!(
        fixture.exists(),
        "repo-root t5-small-f16.gguf must exist for the encoder-decoder gate"
    );

    let engine = block_on(CogentEngine::load(&fixture, NativeRuntimeConfig::default()))
        .expect("load t5-small-f16.gguf");

    let state = block_on(engine.state()).expect("engine state");
    let capabilities = state
        .model
        .as_ref()
        .map(|model| &model.capabilities)
        .expect("model state on loaded engine");
    assert_eq!(capabilities.model_class, ModelClass::EncoderDecoder);
    assert!(capabilities.supports_text_generation);
    assert!(!capabilities.supports_embeddings);
    assert!(
        !capabilities.has_chat_template,
        "T5 has no chat template; chat() must be rejected"
    );

    // chat() must reject before touching the runtime.
    let chat_error = block_on(engine.chat(ChatRequest::new(vec![ChatMessage::new(
        ChatRole::User,
        "hello",
    )])))
    .expect_err("chat() must reject T5");
    assert!(
        matches!(
            &chat_error,
            Error::UnsupportedOperation {
                operation: "chat",
                ..
            }
        ),
        "expected UnsupportedOperation; got: {chat_error:?}"
    );

    // query() against T5 should run the encoder pre-pass + decoder loop and
    // return a non-empty text result.
    let result = block_on(engine.query(QueryRequest::new(
        "translate English to German: Hello, world.",
    )))
    .expect("T5 query");
    assert!(
        !result.text.is_empty(),
        "T5 encoder-decoder query produced empty output"
    );
}
