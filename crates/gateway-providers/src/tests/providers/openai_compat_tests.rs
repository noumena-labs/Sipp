//! Tests the `providers::openai_compat` module in `cogentlm-gateway-providers`.
//!
//! Covers OpenAI-compatible request mapping, response parsing, usage and finish
//! normalization, embedding validation, and SSE stream parsing with deterministic
//! JSON/byte fixtures and no live network calls.

use bytes::Bytes;
use futures_util::StreamExt;

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

    let response = openai_chat_response_from_body(
        Some("req-1".to_string()),
        body,
        ProviderKind::OpenAiCompatible,
    )
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
fn rejects_non_object_openai_compatible_usage() {
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "model": "model-a",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "hi"
            },
            "finish_reason": "stop"
        }],
        "usage": []
    });

    let err = openai_chat_response_from_body(None, body, ProviderKind::OpenAiCompatible)
        .expect_err("array usage should fail");

    assert_eq!(err.kind, ProviderErrorKind::Provider);
    assert_eq!(err.message, "usage must be a JSON object");
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
    let err = openai_chat_body(&request, ProviderKind::OpenAiCompatible)
        .expect_err("zero max_tokens fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let request = chat_request(
        ProviderGenerationOptions {
            temperature: Some(f32::NAN),
            ..ProviderGenerationOptions::default()
        },
        vec![ChatMessage::new(ChatRole::User, "hello")],
    );
    let err = openai_chat_body(&request, ProviderKind::OpenAiCompatible)
        .expect_err("non-finite temp fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let request = chat_request(
        ProviderGenerationOptions {
            temperature: Some(-0.1),
            ..ProviderGenerationOptions::default()
        },
        vec![ChatMessage::new(ChatRole::User, "hello")],
    );
    let err = openai_chat_body(&request, ProviderKind::OpenAiCompatible)
        .expect_err("negative temp fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert_eq!(
        err.message,
        "temperature must be greater than or equal to zero"
    );

    let request = chat_request(
        ProviderGenerationOptions {
            top_p: Some(1.1),
            ..ProviderGenerationOptions::default()
        },
        vec![ChatMessage::new(ChatRole::User, "hello")],
    );
    let err =
        openai_chat_body(&request, ProviderKind::OpenAiCompatible).expect_err("high top_p fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert_eq!(err.message, "top_p must be between 0 and 1");

    let request = chat_request(ProviderGenerationOptions::default(), Vec::new());
    let err = openai_chat_body(&request, ProviderKind::OpenAiCompatible)
        .expect_err("empty messages fail");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn maps_openai_compatible_chat_body_roles_options_stop_and_provider_options() {
    let mut request = chat_request(
        ProviderGenerationOptions {
            max_tokens: Some(16),
            temperature: Some(0.25),
            top_p: Some(0.9),
            stop: vec!["END".to_string()],
        },
        vec![
            ChatMessage::new(ChatRole::System, "system"),
            ChatMessage::new(ChatRole::User, "hello"),
            ChatMessage::new(ChatRole::Assistant, "hi"),
        ],
    );
    request
        .provider_options
        .insert("seed".to_string(), serde_json::json!(7));

    let body = openai_chat_body(&request, ProviderKind::OpenAiCompatible).expect("chat body");
    let stream_body =
        openai_stream_chat_body(&request, ProviderKind::OpenAiCompatible).expect("stream body");

    assert_eq!(body["model"], "model-a");
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][2]["role"], "assistant");
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(body["temperature"], serde_json::json!(0.25));
    assert_eq!(body["top_p"], serde_json::json!(0.9_f32));
    assert_eq!(body["stop"], serde_json::json!(["END"]));
    assert_eq!(body["seed"], 7);
    assert_eq!(stream_body["stream"], true);
}

#[test]
fn maps_openai_embedding_body_and_rejects_invalid_inputs() {
    let mut request = ProviderEmbedRequest {
        model: "embed-model".to_string(),
        input: "hello".to_string(),
        provider_options: Default::default(),
    };
    request
        .provider_options
        .insert("dimensions".to_string(), serde_json::json!(64));

    let body =
        openai_embedding_body(&request, ProviderKind::OpenAiCompatible).expect("embedding body");
    assert_eq!(body["model"], "embed-model");
    assert_eq!(body["input"], "hello");
    assert_eq!(body["encoding_format"], "float");
    assert_eq!(body["dimensions"], 64);

    let mut invalid = request.clone();
    invalid.model = " ".to_string();
    let err = openai_embedding_body(&invalid, ProviderKind::OpenAiCompatible)
        .expect_err("blank model should fail");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let mut invalid = request.clone();
    invalid.input = " ".to_string();
    let err = openai_embedding_body(&invalid, ProviderKind::OpenAiCompatible)
        .expect_err("blank input should fail");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let mut invalid = request;
    invalid
        .provider_options
        .insert("encoding_format".to_string(), serde_json::json!("base64"));
    let err = openai_embedding_body(&invalid, ProviderKind::OpenAiCompatible)
        .expect_err("typed option collision should fail");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn rejects_malformed_openai_model_responses() {
    let err = openai_models_from_body(
        &serde_json::json!({ "data": null }),
        ProviderKind::OpenAiCompatible,
    )
    .expect_err("missing data array");
    assert_eq!(err.kind, ProviderErrorKind::Provider);

    let err = openai_model_from_value(
        &serde_json::json!({ "object": "model" }),
        ProviderKind::OpenAiCompatible,
    )
    .expect_err("missing id");
    assert_eq!(err.kind, ProviderErrorKind::Provider);
}

#[test]
fn maps_openai_chat_response_usage_and_length_finish() {
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "model": "model-a",
        "choices": [{
            "message": { "role": "assistant", "content": "done" },
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5
        }
    });

    let response = openai_chat_response_from_body(
        Some("req-1".to_string()),
        body,
        ProviderKind::OpenAiCompatible,
    )
    .expect("chat response");

    assert_eq!(response.result.text, "done");
    assert_eq!(response.result.finish_reason, FinishReason::Length);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(5));
    assert_eq!(response.metadata.response_id.as_deref(), Some("chatcmpl-1"));
}

#[test]
fn rejects_malformed_openai_chat_responses() {
    let err = openai_chat_response_from_body(
        None,
        serde_json::json!({ "model": "model-a", "choices": [] }),
        ProviderKind::OpenAiCompatible,
    )
    .expect_err("missing first choice");
    assert_eq!(err.kind, ProviderErrorKind::Provider);

    let err = openai_chat_response_from_body(
        None,
        serde_json::json!({
            "choices": [{ "message": { "content": "hi" }, "finish_reason": "stop" }]
        }),
        ProviderKind::OpenAiCompatible,
    )
    .expect_err("missing model");
    assert_eq!(err.kind, ProviderErrorKind::Provider);

    let err = openai_chat_response_from_body(
        None,
        serde_json::json!({
            "model": "model-a",
            "choices": [{ "message": { "content": "hi" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": "bad" }
        }),
        ProviderKind::OpenAiCompatible,
    )
    .expect_err("invalid usage");
    assert_eq!(err.kind, ProviderErrorKind::Provider);
}

#[test]
fn maps_and_rejects_openai_embedding_responses() {
    let body = serde_json::json!({
        "id": "emb-1",
        "model": "embed-model",
        "data": [{ "embedding": [0.125, -0.5] }],
        "usage": { "prompt_tokens": 2, "total_tokens": 2 }
    });
    let response = openai_embedding_response_from_body(
        Some("req-1".to_string()),
        body,
        ProviderKind::OpenAiCompatible,
    )
    .expect("embedding response");
    assert_eq!(response.result.values, vec![0.125, -0.5]);
    assert_eq!(response.usage.expect("usage").input_tokens, Some(2));
    assert_eq!(response.metadata.response_id.as_deref(), Some("emb-1"));

    let err = openai_embedding_response_from_body(
        None,
        serde_json::json!({ "error": { "message": "quota", "code": "insufficient_quota" } }),
        ProviderKind::OpenAiCompatible,
    )
    .expect_err("body error");
    assert_eq!(err.kind, ProviderErrorKind::QuotaExceeded);

    for body in [
        serde_json::json!({ "model": "embed-model" }),
        serde_json::json!({ "model": "embed-model", "data": [] }),
        serde_json::json!({ "model": "embed-model", "data": [{}] }),
        serde_json::json!({ "model": "embed-model", "data": [{ "embedding": ["bad"] }] }),
        serde_json::json!({ "model": "embed-model", "data": [{ "embedding": [1.0e40] }] }),
        serde_json::json!({ "data": [{ "embedding": [0.0] }] }),
    ] {
        let err = openai_embedding_response_from_body(None, body, ProviderKind::OpenAiCompatible)
            .expect_err("malformed embedding body should fail");
        assert_eq!(err.kind, ProviderErrorKind::Provider);
    }
}

#[test]
fn sse_parser_handles_partial_events() {
    let mut parser = SseParser::new(ProviderKind::OpenAiCompatible);

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
    let mut parser = SseParser::new(ProviderKind::OpenAiCompatible);

    let pushed = parser
        .push(br#"data: {"choices":[{"delta":{"content":"he"}}]}"#)
        .expect("partial push");
    assert!(pushed.is_empty());

    assert_eq!(
        parser.finish().expect("flush trailing event"),
        vec![r#"{"choices":[{"delta":{"content":"he"}}]}"#.to_string()]
    );
}

#[tokio::test]
async fn openai_stream_events_flush_trailing_payload_and_map_length_finish() {
    let stream = openai_stream_events(
        Some("req-stream".to_string()),
        byte_stream(vec![Ok(Bytes::from_static(
            br#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"length"}]}"#,
        ))]),
        ProviderKind::OpenAiCompatible,
    );

    let events = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<ProviderResult<Vec<_>>>()
        .expect("events");

    assert!(matches!(
        &events[0],
        ProviderStreamEvent::TokenBatch(batch)
            if batch.request_id == "req-stream" && batch.text == "hi"
    ));
    assert_eq!(
        events[1],
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Length
        }
    );
}

#[tokio::test]
async fn openai_stream_events_surface_invalid_json_and_byte_errors() {
    let mut stream = openai_stream_events(
        None,
        byte_stream(vec![Ok(Bytes::from_static(b"data: {bad json}\n\n"))]),
        ProviderKind::OpenAiCompatible,
    );
    let err = stream
        .next()
        .await
        .expect("event")
        .expect_err("invalid json should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);

    let mut stream = openai_stream_events(
        None,
        byte_stream(vec![Err(ProviderError::new(
            ProviderErrorKind::Transport,
            ProviderKind::OpenAiCompatible,
            "broken stream",
        ))]),
        ProviderKind::OpenAiCompatible,
    );
    let err = stream
        .next()
        .await
        .expect("event")
        .expect_err("byte error should fail");
    assert_eq!(err.kind, ProviderErrorKind::Transport);
}

#[tokio::test]
async fn openai_stream_events_surface_trailing_invalid_json_at_stream_end() {
    let mut stream = openai_stream_events(
        None,
        byte_stream(vec![Ok(Bytes::from_static(b"data: {bad json}"))]),
        ProviderKind::OpenAiCompatible,
    );

    let err = stream
        .next()
        .await
        .expect("event")
        .expect_err("trailing invalid json should fail");

    assert_eq!(err.kind, ProviderErrorKind::Provider);
}

#[tokio::test]
async fn openai_stream_events_emit_usage_and_ignore_payloads_without_choices() {
    let stream = openai_stream_events(
        None,
        byte_stream(vec![Ok(Bytes::from_static(concat!(
            "data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
            "data: {\"id\":\"chunk-without-choices\"}\n\n",
            "data: [DONE]\n\n"
        ).as_bytes()))]),
        ProviderKind::OpenAiCompatible,
    );

    let events = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<ProviderResult<Vec<_>>>()
        .expect("events");

    assert_eq!(events.len(), 2);
    assert!(matches!(
        events[0],
        ProviderStreamEvent::Usage {
            usage: TokenUsage {
                total_tokens: Some(3),
                ..
            }
        }
    ));
    assert_eq!(
        events[1],
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    );
}

fn byte_stream(items: Vec<ProviderResult<Bytes>>) -> HttpByteStream {
    Box::pin(futures_util::stream::iter(items))
}
