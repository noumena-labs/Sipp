//! Unit tests for the parent module.

use super::ChatRole;

#[test]
fn chat_role_serde_uses_snake_case() {
    let encoded = serde_json::to_string(&ChatRole::System).expect("serialize role");
    assert_eq!(encoded, "\"system\"");

    let decoded: ChatRole = serde_json::from_str("\"assistant\"").expect("deserialize role");
    assert_eq!(decoded, ChatRole::Assistant);
}
