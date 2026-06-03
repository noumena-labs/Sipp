//! Unit tests for the remote gateway transport.

use std::time::Duration;

use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenBatch};
use futures_util::StreamExt;
use reqwest::header::AUTHORIZATION;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::{
    GatewayChatRequest, GatewayConfig, GatewayEmbedRequest, GatewayError, GatewayErrorKind,
    GatewayGenerationOptions, GatewayOptions, GatewayQueryRequest, GatewayResult, GatewaySecret,
    GatewayStreamEvent, GatewayTransport, TokenUsage,
};

use crate::transport::build_headers;

fn config(server: &MockServer) -> GatewayConfig {
    GatewayConfig {
        base_url: server.uri(),
        token: GatewaySecret::new("gateway-token"),
        timeout: None,
    }
}

fn transport(server: &MockServer) -> GatewayTransport {
    GatewayTransport::new(config(server)).expect("gateway transport")
}

fn query_request(model: &str, prompt: &str) -> GatewayQueryRequest {
    GatewayQueryRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        options: GatewayGenerationOptions::default(),
        gateway_options: GatewayOptions::new(),
    }
}

fn chat_request(model: &str) -> GatewayChatRequest {
    GatewayChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage::new(ChatRole::System, "be concise"),
            ChatMessage::new(ChatRole::User, "hello"),
        ],
        options: GatewayGenerationOptions::default(),
        gateway_options: GatewayOptions::new(),
    }
}

fn embed_request(model: &str, input: &str) -> GatewayEmbedRequest {
    GatewayEmbedRequest {
        model: model.to_string(),
        input: input.to_string(),
        gateway_options: GatewayOptions::new(),
    }
}

fn new_error(config: GatewayConfig) -> GatewayError {
    match GatewayTransport::new(config) {
        Ok(_) => panic!("transport should reject config"),
        Err(error) => error,
    }
}

fn collect_events(
    events: Vec<GatewayResult<GatewayStreamEvent>>,
) -> GatewayResult<Vec<GatewayStreamEvent>> {
    events.into_iter().collect()
}

#[test]
fn gateway_secret_debug_redacts_token() {
    let config = GatewayConfig {
        base_url: "https://gateway.example".to_string(),
        token: GatewaySecret::new("secret-token"),
        timeout: None,
    };

    let debug = format!("{config:?}");

    assert!(!debug.contains("secret-token"));
    assert!(debug.contains("[redacted]"));
}

#[test]
fn bearer_header_is_marked_sensitive() {
    let headers = build_headers("secret-token").expect("headers");
    let value = headers.get(AUTHORIZATION).expect("authorization header");

    assert_eq!(value.to_str().expect("header value"), "Bearer secret-token");
    assert!(value.is_sensitive());
}

#[test]
fn validates_gateway_transport_config() {
    let invalid_url = new_error(GatewayConfig {
        base_url: String::new(),
        token: GatewaySecret::new("token"),
        timeout: None,
    });
    assert_eq!(invalid_url.kind, GatewayErrorKind::InvalidRequest);

    let insecure_remote = new_error(GatewayConfig {
        base_url: "http://gateway.example".to_string(),
        token: GatewaySecret::new("token"),
        timeout: None,
    });
    assert_eq!(insecure_remote.kind, GatewayErrorKind::InvalidRequest);

    let userinfo = new_error(GatewayConfig {
        base_url: "https://user:gateway-secret@gateway.example".to_string(),
        token: GatewaySecret::new("token"),
        timeout: None,
    });
    assert_eq!(userinfo.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        userinfo.message,
        "gateway base_url must not include userinfo"
    );
    assert!(!format!("{userinfo:?}").contains("gateway-secret"));

    let empty_token = new_error(GatewayConfig {
        base_url: "https://gateway.example".to_string(),
        token: GatewaySecret::new(""),
        timeout: None,
    });
    assert_eq!(empty_token.kind, GatewayErrorKind::Authentication);

    let zero_timeout = new_error(GatewayConfig {
        base_url: "https://gateway.example".to_string(),
        token: GatewaySecret::new("token"),
        timeout: Some(Duration::ZERO),
    });
    assert_eq!(zero_timeout.kind, GatewayErrorKind::InvalidRequest);

    assert!(GatewayTransport::new(GatewayConfig {
        base_url: "http://localhost:8080".to_string(),
        token: GatewaySecret::new("token"),
        timeout: None,
    })
    .is_ok());
    assert!(GatewayTransport::new(GatewayConfig {
        base_url: "http://127.0.0.1:8080".to_string(),
        token: GatewaySecret::new("token"),
        timeout: None,
    })
    .is_ok());
    assert!(GatewayTransport::new(GatewayConfig {
        base_url: "http://127.10.0.1:8080".to_string(),
        token: GatewaySecret::new("token"),
        timeout: None,
    })
    .is_ok());
    assert!(GatewayTransport::new(GatewayConfig {
        base_url: "http://[::1]:8080".to_string(),
        token: GatewaySecret::new("token"),
        timeout: None,
    })
    .is_ok());
}

#[tokio::test]
async fn query_posts_to_gateway_protocol_with_bearer_auth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .and(header("authorization", "Bearer gateway-token"))
        .and(body_json(json!({
            "model": "local-seq2seq",
            "prompt": "hello",
            "max_tokens": 16,
            "stop": ["END"],
            "stream": false,
            "trace": "public"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-query")
                .set_body_json(json!({
                    "id": "resp-query",
                    "model": "local-seq2seq",
                    "text": "world",
                    "finish_reason": "max_tokens",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3
                    }
                })),
        )
        .mount(&server)
        .await;

    let mut req = query_request("local-seq2seq", "hello");
    req.options.max_tokens = Some(16);
    req.options.stop.push("END".to_string());
    req.gateway_options
        .insert("trace".to_string(), json!("public"));

    let response = transport(&server).query(req).await.expect("query response");

    assert_eq!(response.result.text, "world");
    assert_eq!(response.result.finish_reason, FinishReason::Length);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(3));
    assert_eq!(response.metadata.request_id.as_deref(), Some("req-query"));
    assert_eq!(response.metadata.response_id.as_deref(), Some("resp-query"));
}

#[tokio::test]
async fn chat_and_embed_post_to_gateway_protocol() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .and(header("authorization", "Bearer gateway-token"))
        .and(body_json(json!({
            "model": "chat-pro",
            "messages": [
                { "role": "system", "content": "be concise" },
                { "role": "user", "content": "hello" }
            ],
            "stream": false
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-chat")
                .set_body_json(json!({
                    "id": "resp-chat",
                    "model": "chat-pro",
                    "text": "hi",
                    "finish_reason": "stop"
                })),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/embed"))
        .and(header("authorization", "Bearer gateway-token"))
        .and(body_json(json!({
            "model": "embed-small",
            "input": "hello",
            "input_type": "query"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-embed")
                .set_body_json(json!({
                    "id": "resp-embed",
                    "model": "embed-small",
                    "embedding": [0.25, -0.5],
                    "usage": {
                        "input_tokens": 1,
                        "total_tokens": 1
                    }
                })),
        )
        .mount(&server)
        .await;

    let gateway = transport(&server);
    let chat = gateway
        .chat(chat_request("chat-pro"))
        .await
        .expect("chat response");
    let mut embed_req = embed_request("embed-small", "hello");
    embed_req
        .gateway_options
        .insert("input_type".to_string(), json!("query"));
    let embed = gateway.embed(embed_req).await.expect("embedding response");

    assert_eq!(chat.result.text, "hi");
    assert_eq!(chat.metadata.request_id.as_deref(), Some("req-chat"));
    assert_eq!(embed.result.values, vec![0.25, -0.5]);
    assert_eq!(embed.metadata.request_id.as_deref(), Some("req-embed"));
    assert_eq!(embed.usage.expect("usage").input_tokens, Some(1));
}

#[tokio::test]
async fn stream_query_parses_gateway_sse_events() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .and(header("authorization", "Bearer gateway-token"))
        .and(body_json(json!({
            "model": "local-seq2seq",
            "prompt": "hello",
            "stream": true
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-stream")
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "event: token\n",
                    "data: {\"text\":\"he\",\"sequence\":7}\n\n",
                    "event: usage\n",
                    "data: {\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}\n\n",
                    "event: token\n",
                    "data: {\"text\":\"llo\"}\n\n",
                    "event: done\n",
                    "data: {\"finish_reason\":\"length\"}\n\n",
                    "data: [DONE]\n\n"
                )),
        )
        .mount(&server)
        .await;

    let events = transport(&server)
        .stream_query(query_request("local-seq2seq", "hello"))
        .await
        .expect("stream")
        .collect::<Vec<_>>()
        .await;
    let events = collect_events(events).expect("events");

    assert!(matches!(
        &events[0],
        GatewayStreamEvent::TokenBatch(TokenBatch {
            request_id,
            text,
            sequence_start,
            ..
        }) if request_id == "req-stream" && text == "he" && *sequence_start == 7
    ));
    assert!(matches!(
        &events[1],
        GatewayStreamEvent::Usage {
            usage: TokenUsage {
                total_tokens: Some(3),
                ..
            }
        }
    ));
    assert!(matches!(
        &events[2],
        GatewayStreamEvent::TokenBatch(TokenBatch {
            request_id,
            text,
            sequence_start,
            ..
        }) if request_id == "req-stream" && text == "llo" && *sequence_start == 8
    ));
    assert!(matches!(
        events[3],
        GatewayStreamEvent::Finished {
            finish_reason: FinishReason::Length
        }
    ));
}

#[tokio::test]
async fn stream_error_preserves_gateway_request_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .and(header("authorization", "Bearer gateway-token"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-stream-error")
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "event: error\n",
                    "data: {\"error\":{\"message\":\"not allowed\",\"code\":\"permission_error\"}}\n\n"
                )),
        )
        .mount(&server)
        .await;

    let mut stream = transport(&server)
        .stream_chat(chat_request("chat-pro"))
        .await
        .expect("stream");
    let err = stream
        .next()
        .await
        .expect("first event")
        .expect_err("stream error event should fail");

    assert_eq!(err.kind, GatewayErrorKind::Authorization);
    assert_eq!(err.code.as_deref(), Some("permission_error"));
    assert_eq!(err.request_id.as_deref(), Some("req-stream-error"));
}

#[tokio::test]
async fn stream_errors_redact_bearer_token_echoes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .and(header("authorization", "Bearer gateway-token"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-gateway-token")
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "event: error\n",
                    "data: {\"error\":{\"message\":\"stream gateway-token\",\"code\":\"permission_gateway-token\",\"details\":[\"gateway-token\"]}}\n\n"
                )),
        )
        .mount(&server)
        .await;

    let mut stream = transport(&server)
        .stream_chat(chat_request("chat-pro"))
        .await
        .expect("stream");
    let err = stream
        .next()
        .await
        .expect("first event")
        .expect_err("stream error event should fail");

    assert_eq!(err.message, "stream [redacted]");
    assert_eq!(err.code.as_deref(), Some("permission_[redacted]"));
    assert_eq!(err.request_id.as_deref(), Some("req-[redacted]"));
    assert!(!format!("{err:?}").contains("gateway-token"));
    assert!(!format!("{:?}", err.raw).contains("gateway-token"));
}

#[tokio::test]
async fn maps_gateway_http_error_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("x-request-id", "req-rate-limit")
                .insert_header("retry-after-ms", "1500")
                .set_body_json(json!({
                    "error": {
                        "message": "slow down",
                        "code": "rate_limit"
                    }
                })),
        )
        .mount(&server)
        .await;

    let err = transport(&server)
        .query(query_request("chat-pro", "hello"))
        .await
        .expect_err("429 should fail");

    assert_eq!(err.kind, GatewayErrorKind::RateLimited);
    assert_eq!(err.status, Some(429));
    assert_eq!(err.code.as_deref(), Some("rate_limit"));
    assert_eq!(err.request_id.as_deref(), Some("req-rate-limit"));
    assert_eq!(err.retry_after, Some(Duration::from_millis(1500)));
}

#[tokio::test]
async fn gateway_http_errors_redact_bearer_token_echoes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .respond_with(
            ResponseTemplate::new(401)
                .insert_header("x-request-id", "req-gateway-token")
                .set_body_json(json!({
                    "error": {
                        "message": "invalid gateway-token",
                        "code": "authentication_gateway-token",
                        "details": ["gateway-token"]
                    }
                })),
        )
        .mount(&server)
        .await;

    let err = transport(&server)
        .query(query_request("chat-pro", "hello"))
        .await
        .expect_err("401 should fail");

    assert_eq!(err.kind, GatewayErrorKind::Authentication);
    assert_eq!(err.message, "invalid [redacted]");
    assert_eq!(err.code.as_deref(), Some("authentication_[redacted]"));
    assert_eq!(err.request_id.as_deref(), Some("req-[redacted]"));
    assert!(!format!("{err:?}").contains("gateway-token"));
    assert!(!format!("{:?}", err.raw).contains("gateway-token"));
}

#[tokio::test]
async fn maps_unsupported_feature_error_code() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "message": "model alias does not support query",
                "code": "unsupported_feature"
            }
        })))
        .mount(&server)
        .await;

    let err = transport(&server)
        .query(query_request("chat-only", "hello"))
        .await
        .expect_err("unsupported feature should fail");

    assert_eq!(err.kind, GatewayErrorKind::UnsupportedFeature);
    assert_eq!(err.status, Some(400));
}

#[tokio::test]
async fn does_not_follow_gateway_redirects() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .respond_with(
            ResponseTemplate::new(307)
                .insert_header("location", format!("{}/redirected", server.uri())),
        )
        .mount(&server)
        .await;

    let err = transport(&server)
        .query(query_request("chat-pro", "hello"))
        .await
        .expect_err("redirect should fail");

    assert_eq!(err.status, Some(307));
    assert_eq!(err.kind, GatewayErrorKind::Gateway);
}

#[tokio::test]
async fn request_validation_rejects_gateway_option_collisions() {
    let server = MockServer::start().await;
    let gateway = transport(&server);
    let mut req = query_request("model", "hello");
    req.gateway_options
        .insert("model".to_string(), json!("override"));

    let err = gateway
        .query(req)
        .await
        .expect_err("typed field collision should fail");

    assert_eq!(err.kind, GatewayErrorKind::InvalidRequest);
}

#[tokio::test]
async fn request_validation_rejects_local_only_gateway_options() {
    let server = MockServer::start().await;
    let gateway = transport(&server);
    let mut query = query_request("model", "hello");
    query
        .gateway_options
        .insert("grammar".to_string(), json!("root ::= \"ok\""));

    let err = gateway
        .query(query)
        .await
        .expect_err("local-only gateway option should fail");

    assert_eq!(err.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        err.message,
        "gateway_options cannot contain local-only field: grammar"
    );

    let mut embed = embed_request("model", "hello");
    embed
        .gateway_options
        .insert("normalize".to_string(), json!(true));

    let err = gateway
        .embed(embed)
        .await
        .expect_err("local-only gateway option should fail");

    assert_eq!(err.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        err.message,
        "gateway_options cannot contain local-only field: normalize"
    );
}
