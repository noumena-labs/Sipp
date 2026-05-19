//! Unit tests for the parent module.

use super::*;
use crate::engine::{GenerateOptions, SamplingRuntimeConfig};

#[test]
fn render_messages_json_uses_template_roles() {
    let messages = vec![
        ChatMessage::system("policy"),
        ChatMessage::user("hello"),
        ChatMessage::assistant("hi"),
    ];

    let rendered = render_messages_json(&messages).expect("json");

    assert_eq!(
        rendered,
        r#"[{"content":"policy","role":"system"},{"content":"hello","role":"user"},{"content":"hi","role":"assistant"}]"#
    );
}

#[test]
fn query_options_default_matches_public_completion_defaults() {
    let options = QueryOptions::default();

    assert_eq!(options.context_key, "default");
    assert_eq!(options.max_tokens, 64);
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
fn query_request_defaults_options() {
    let request = QueryRequest::new("hello");

    assert_eq!(request.prompt, "hello");
    assert_eq!(request.options, QueryOptions::default());
}

#[test]
fn emit_event_drops_closed_subscribers() {
    let subscribers = Arc::new(Mutex::new(Vec::new()));
    let (closed_tx, closed_rx) = mpsc::channel();
    drop(closed_rx);
    let (open_tx, open_rx) = mpsc::channel();
    subscribers.lock().unwrap().push(closed_tx);
    subscribers.lock().unwrap().push(open_tx);

    emit_event(&subscribers, EngineEvent::Closed);

    assert!(matches!(open_rx.recv().unwrap(), EngineEvent::Closed));
    assert_eq!(subscribers.lock().unwrap().len(), 1);
}

#[test]
fn engine_handle_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<CogentEngine>();
}
