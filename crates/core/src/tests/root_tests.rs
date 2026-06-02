//! Tests the `cogentlm-core` crate root public value types.
//!
//! Covers exported chat roles, finish reasons, capability support, token usage,
//! and token emission values with deterministic model-free assertions.

use super::{
    CapabilitySupport, ChatMessage, ChatRole, FinishReason, TokenBatch, TokenEmissionStats,
    TokenUsage,
};

#[test]
fn chat_role_as_str_and_serde_cover_all_variants() {
    let cases = [
        (ChatRole::System, "system"),
        (ChatRole::User, "user"),
        (ChatRole::Assistant, "assistant"),
    ];

    for (role, expected) in cases {
        assert_eq!(role.as_str(), expected);

        let encoded = serde_json::to_string(&role).expect("serialize role");
        assert_eq!(encoded, format!("\"{expected}\""));

        let decoded: ChatRole = serde_json::from_str(&encoded).expect("deserialize role");
        assert_eq!(decoded, role);
    }
}

#[test]
fn chat_message_new_preserves_role_and_content() {
    let borrowed = ChatMessage::new(ChatRole::User, "hello");
    assert_eq!(
        borrowed,
        ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        }
    );

    let owned = ChatMessage::new(ChatRole::Assistant, String::from("answer"));
    assert_eq!(owned.role, ChatRole::Assistant);
    assert_eq!(owned.content, "answer");
}

#[test]
fn finish_reason_as_str_covers_all_variants() {
    let cases = [
        (FinishReason::Stop, "stop"),
        (FinishReason::Length, "length"),
        (FinishReason::Cancelled, "cancelled"),
        (FinishReason::Error, "error"),
    ];

    for (reason, expected) in cases {
        assert_eq!(reason.as_str(), expected);
    }
}

#[test]
fn capability_support_variants_remain_distinct_copy_values() {
    let supported = CapabilitySupport::Supported;
    let copied = supported;
    assert_eq!(supported, copied);

    assert_ne!(CapabilitySupport::Supported, CapabilitySupport::Unsupported);
    assert_ne!(CapabilitySupport::Supported, CapabilitySupport::Unknown);
    assert_ne!(CapabilitySupport::Unsupported, CapabilitySupport::Unknown);

    assert_eq!(format!("{:?}", CapabilitySupport::Supported), "Supported");
    assert_eq!(
        format!("{:?}", CapabilitySupport::Unsupported),
        "Unsupported"
    );
    assert_eq!(format!("{:?}", CapabilitySupport::Unknown), "Unknown");
}

#[test]
fn token_usage_default_and_partial_counts_preserve_fields() {
    assert_eq!(
        TokenUsage::default(),
        TokenUsage {
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
        }
    );

    let usage = TokenUsage {
        input_tokens: Some(3),
        output_tokens: None,
        total_tokens: Some(3),
    };

    assert_eq!(usage.input_tokens, Some(3));
    assert_eq!(usage.output_tokens, None);
    assert_eq!(usage.total_tokens, Some(3));
}

#[test]
fn token_emission_stats_default_and_batch_preserve_fields() {
    assert_eq!(
        TokenEmissionStats::default(),
        TokenEmissionStats {
            frames_sent: 0,
            bytes_sent: 0,
            batches_sent: 0,
        }
    );

    let batch = TokenBatch {
        request_id: "request-1".to_string(),
        stream_id: 7,
        sequence_start: 11,
        text: "hello".to_string(),
        frame_count: 2,
        byte_count: 5,
        stats: TokenEmissionStats {
            frames_sent: 9,
            bytes_sent: 12,
            batches_sent: 3,
        },
    };
    let cloned = batch.clone();

    assert_eq!(cloned, batch);
    assert_eq!(batch.request_id, "request-1");
    assert_eq!(batch.stream_id, 7);
    assert_eq!(batch.sequence_start, 11);
    assert_eq!(batch.text, "hello");
    assert_eq!(batch.frame_count, 2);
    assert_eq!(batch.byte_count, 5);
    assert_eq!(batch.stats.frames_sent, 9);
    assert_eq!(batch.stats.bytes_sent, 12);
    assert_eq!(batch.stats.batches_sent, 3);
}
