//! Unit tests for the parent module.

use super::{ChatMessage, ChatRole, FinishReason, TokenBatch, TokenEmissionStats};

#[test]
fn shared_core_types_reexport_from_engine() {
    let message = ChatMessage::new(ChatRole::User, "hello");
    assert_eq!(message.role.as_str(), "user");

    let batch = TokenBatch {
        request_id: "request".to_string(),
        stream_id: 1,
        sequence_start: 0,
        text: "hello".to_string(),
        frame_count: 1,
        byte_count: 5,
        stats: TokenEmissionStats::default(),
    };
    assert_eq!(batch.text, "hello");
    assert_eq!(FinishReason::Stop.as_str(), "stop");
}
