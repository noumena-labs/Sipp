//! Unit tests for the parent module.

use super::*;
use crate::ProviderGenerationOptions;

fn chat_request(
    options: ProviderGenerationOptions,
    messages: Vec<ChatMessage>,
) -> ProviderChatRequest {
    ProviderChatRequest {
        model: "model-a".to_string(),
        messages,
        options,
        provider_options: Default::default(),
    }
}

#[test]
fn tool_call_response_yields_empty_text_with_raw_preserved() {
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "model": "model-a",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let response =
        openai_chat_response_from_body(Some("req-1".to_string()), body, ProviderKind::Proxy)
            .expect("tool-call response should parse");

    assert_eq!(response.result.text, "");
    assert_eq!(
        response.metadata.finish_reason_raw.as_deref(),
        Some("tool_calls")
    );
    assert!(response
        .metadata
        .raw
        .pointer("/choices/0/message/tool_calls")
        .is_some());
}

#[test]
fn rejects_invalid_openai_compatible_request_options() {
    let request = chat_request(
        ProviderGenerationOptions {
            max_tokens: Some(0),
            ..ProviderGenerationOptions::default()
        },
        vec![ChatMessage::new(ChatRole::User, "hello")],
    );
    let err = openai_chat_body(&request, ProviderKind::Proxy).expect_err("zero max_tokens fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let request = chat_request(
        ProviderGenerationOptions {
            temperature: Some(f32::NAN),
            ..ProviderGenerationOptions::default()
        },
        vec![ChatMessage::new(ChatRole::User, "hello")],
    );
    let err = openai_chat_body(&request, ProviderKind::Proxy).expect_err("non-finite temp fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let request = chat_request(ProviderGenerationOptions::default(), Vec::new());
    let err = openai_chat_body(&request, ProviderKind::Proxy).expect_err("empty messages fail");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn sse_parser_handles_partial_events() {
    let mut parser = SseParser::new(ProviderKind::Proxy);

    let first = parser
        .push(br#"data: {"choices":[{"delta":{"content":"he"}"#)
        .expect("partial push");
    assert!(first.is_empty());

    let second = parser
        .push(b"}]}\n\ndata: [DONE]\n\n")
        .expect("complete push");
    assert_eq!(
        second,
        vec![
            r#"{"choices":[{"delta":{"content":"he"}}]}"#.to_string(),
            "[DONE]".to_string()
        ]
    );
}

#[test]
fn sse_parser_flushes_trailing_event() {
    let mut parser = SseParser::new(ProviderKind::Proxy);

    let pushed = parser
        .push(br#"data: {"choices":[{"delta":{"content":"he"}}]}"#)
        .expect("partial push");
    assert!(pushed.is_empty());

    assert_eq!(
        parser.finish().expect("flush trailing event"),
        vec![r#"{"choices":[{"delta":{"content":"he"}}]}"#.to_string()]
    );
}
