//! Unit tests for the parent module.

use std::sync::{Arc, Mutex};

use super::super::{render_messages_json, start_chat, ChatMessage, ChatRequest, ChatRole};
use crate::error::Error;
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::tests::runtime_tests::test_runtime;

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
    let error = match start_chat(&mut runtime, request, &subscribers) {
        Err(error) => error,
        Ok(_) => panic!("chat() must reject when has_chat_template is false"),
    };

    assert!(
        matches!(&error, Error::UnsupportedOperation { operation: "chat", reason }
            if reason.contains("no chat template")),
        "expected chat() to reject with UnsupportedOperation; got: {error:?}"
    );
}
