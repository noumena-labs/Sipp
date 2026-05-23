//! Unit tests for the parent module.

use super::super::{render_messages_json, ChatMessage, ChatRole};

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
