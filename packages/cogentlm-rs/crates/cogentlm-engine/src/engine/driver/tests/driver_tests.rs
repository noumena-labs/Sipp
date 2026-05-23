//! Unit tests for the parent module.

use super::super::*;
use crate::engine::{
    GenerateOptions, SamplingRuntimeConfig, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};

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
    assert_eq!(
        options
            .sampling
            .as_ref()
            .and_then(|sampling| sampling.temperature),
        Some(0.1)
    );
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
fn chat_role_choice_helper_accepts_public_roles() {
    assert_eq!(ChatRole::from_choice("system"), Some(ChatRole::System));
    assert_eq!(
        ChatRole::from_choice(" ASSISTANT "),
        Some(ChatRole::Assistant)
    );
    assert_eq!(
        ChatRole::from_choice("assistant"),
        Some(ChatRole::Assistant)
    );
}

#[test]
fn engine_handle_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<CogentEngine>();
}
