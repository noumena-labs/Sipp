//! Tests the `remote_endpoint` module in `cogentlm-client`.
//!
//! Covers remote request mapping, streaming, gateway error propagation, and
//! task lifecycle behavior through deterministic gateway fixtures.

use std::future::Future;

use cogentlm_core::{CapabilitySupport, ChatMessage, ChatRole, FinishReason};
use cogentlm_remote::{GatewayConfig, GatewaySecret, GatewayTransport};
use futures::executor::block_on as sync_block_on;
use futures::StreamExt;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::dispatch::InferenceEndpoint;
use crate::{CogentTextOptions, LocalEmbedOptions, LocalTextOptions, RemoteErrorKind};

fn run_async<T>(future: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(future)
}

fn gateway_transport(base_url: String, token: &str) -> GatewayTransport {
    GatewayTransport::new(GatewayConfig {
        base_url,
        token: GatewaySecret::new(token),
        timeout: None,
    })
    .expect("gateway transport")
}

fn endpoint(server: &MockServer) -> RemoteEndpoint {
    RemoteEndpoint::new(
        EndpointRef::Remote {
            id: "remote".to_string(),
        },
        "remote-model".to_string(),
        EndpointCapabilities::unknown(),
        gateway_transport(server.uri(), "gateway-token"),
        RemoteExecutor::new().expect("remote executor"),
    )
}

#[test]
fn remote_generation_options_preserve_common_text_fields() {
    let options = remote_generation_options(CogentTextOptions {
        max_tokens: Some(7),
        temperature: Some(0.25),
        top_p: Some(0.9),
        stop: vec!["stop".to_string()],
    });

    assert_eq!(options.max_tokens, Some(7));
    assert_eq!(options.temperature, Some(0.25));
    assert_eq!(options.top_p, Some(0.9));
    assert_eq!(options.stop, vec!["stop"]);
}

#[test]
fn capabilities_returns_configured_capability_snapshot() {
    let capabilities = EndpointCapabilities {
        query: CapabilitySupport::Supported,
        chat: CapabilitySupport::Unknown,
        embed: CapabilitySupport::Unsupported,
    };
    let endpoint = RemoteEndpoint::new(
        EndpointRef::Remote {
            id: "remote".to_string(),
        },
        "remote-model".to_string(),
        capabilities.clone(),
        gateway_transport("http://localhost:11434".to_string(), "gateway-token"),
        RemoteExecutor::new().expect("remote executor"),
    );

    assert_eq!(endpoint.capabilities(), &capabilities);
}

#[test]
fn query_maps_gateway_response_to_client_response() {
    run_async(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/query"))
            .and(header("authorization", "Bearer gateway-token"))
            .and(body_json(json!({
                "model": "remote-model",
                "prompt": "hello",
                "max_tokens": 3,
                "stream": false
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-request-id", "req-query")
                    .set_body_json(json!({
                        "id": "resp-query",
                        "model": "remote-model",
                        "text": "echo:hello",
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2
                        }
                    })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let endpoint = endpoint(&server);
        let response = endpoint
            .query(CogentQueryRequest {
                prompt: "hello".to_string(),
                options: CogentTextOptions {
                    max_tokens: Some(3),
                    ..CogentTextOptions::default()
                },
                ..CogentQueryRequest::default()
            })
            .await
            .expect("query response");

        assert_eq!(response.text, "echo:hello");
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.endpoint, *endpoint.endpoint());
        assert_eq!(response.usage.expect("usage").total_tokens, Some(2));
        assert!(response.local_stats.is_none());
    });
}

#[test]
fn chat_maps_gateway_response_to_client_response() {
    run_async(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat"))
            .and(header("authorization", "Bearer gateway-token"))
            .and(body_json(json!({
                "model": "remote-model",
                "messages": [{ "role": "user", "content": "hello" }],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp-chat",
                "model": "remote-model",
                "text": "hi",
                "finish_reason": "length"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let endpoint = endpoint(&server);
        let response = endpoint
            .chat(CogentChatRequest {
                messages: vec![ChatMessage::new(ChatRole::User, "hello")],
                ..CogentChatRequest::default()
            })
            .await
            .expect("chat response");

        assert_eq!(response.text, "hi");
        assert_eq!(response.finish_reason, FinishReason::Length);
        assert_eq!(response.endpoint, *endpoint.endpoint());
    });
}

#[test]
fn embed_maps_gateway_response_to_client_response() {
    run_async(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embed"))
            .and(header("authorization", "Bearer gateway-token"))
            .and(body_json(json!({
                "model": "remote-model",
                "input": "abc"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp-embed",
                "model": "remote-model",
                "embedding": [1.0, 2.0, 3.0],
                "usage": {
                    "input_tokens": 3,
                    "total_tokens": 3
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let endpoint = endpoint(&server);
        let response = endpoint
            .embed(CogentEmbedRequest {
                input: "abc".to_string(),
                ..CogentEmbedRequest::default()
            })
            .await
            .expect("embed response");

        assert_eq!(response.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(response.endpoint, *endpoint.endpoint());
        assert_eq!(response.usage.expect("usage").input_tokens, Some(3));
        assert!(response.local_stats.is_none());
        assert!(response.pooling.is_none());
        assert!(response.normalized.is_none());
    });
}

#[test]
fn query_stream_forwards_token_batches_and_final_response() {
    run_async(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/query"))
            .and(header("authorization", "Bearer gateway-token"))
            .and(body_json(json!({
                "model": "remote-model",
                "prompt": "hello",
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-request-id", "req-stream")
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(concat!(
                        "event: token\n",
                        "data: {\"text\":\"a\"}\n\n",
                        "event: token\n",
                        "data: {\"text\":\"b\"}\n\n",
                        "event: usage\n",
                        "data: {\"input_tokens\":2,\"output_tokens\":2,\"total_tokens\":4}\n\n",
                        "event: done\n",
                        "data: {\"finish_reason\":\"length\"}\n\n",
                        "data: [DONE]\n\n"
                    )),
            )
            .expect(1)
            .mount(&server)
            .await;

        let endpoint = endpoint(&server);
        let run = endpoint.query(CogentQueryRequest {
            prompt: "hello".to_string(),
            emit_tokens: true,
            ..CogentQueryRequest::default()
        });
        let (tokens, response) = run.into_parts();
        let response = response.await.expect("query stream response");
        let tokens = tokens.collect::<Vec<_>>().await;

        assert_eq!(response.text, "ab");
        assert_eq!(response.finish_reason, FinishReason::Length);
        assert_eq!(response.usage.expect("usage").total_tokens, Some(4));
        assert_eq!(
            tokens
                .iter()
                .map(|batch| batch.text.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert!(tokens.iter().all(|batch| batch.request_id == "req-stream"));
    });
}

#[test]
fn gateway_errors_are_mapped_for_query_chat_and_embed() {
    run_async(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/query"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {
                    "message": "limited",
                    "code": "rate_limit_error"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "error": {
                    "message": "busy",
                    "code": "overloaded_error"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/embed"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {
                    "message": "bad",
                    "code": "invalid_request"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let endpoint = endpoint(&server);
        let query_error = endpoint
            .query(CogentQueryRequest {
                prompt: "hello".to_string(),
                ..CogentQueryRequest::default()
            })
            .await
            .expect_err("query gateway error");
        let chat_error = endpoint
            .chat(CogentChatRequest {
                messages: vec![ChatMessage::new(ChatRole::User, "hello")],
                ..CogentChatRequest::default()
            })
            .await
            .expect_err("chat gateway error");
        let embed_error = endpoint
            .embed(CogentEmbedRequest {
                input: "abc".to_string(),
                ..CogentEmbedRequest::default()
            })
            .await
            .expect_err("embed gateway error");

        assert!(matches!(
            query_error,
            CogentError::Remote(remote) if remote.kind == RemoteErrorKind::RateLimited
        ));
        assert!(matches!(
            chat_error,
            CogentError::Remote(remote) if remote.kind == RemoteErrorKind::Overloaded
        ));
        assert!(matches!(
            embed_error,
            CogentError::Remote(remote) if remote.kind == RemoteErrorKind::InvalidRequest
        ));
    });
}

#[test]
fn validation_rejects_local_options_before_http_dispatch() {
    run_async(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let endpoint = endpoint(&server);
        let query_error = endpoint
            .query(CogentQueryRequest {
                prompt: "hello".to_string(),
                local: LocalTextOptions {
                    grammar: Some("root ::= \"ok\"".to_string()),
                    ..LocalTextOptions::default()
                },
                ..CogentQueryRequest::default()
            })
            .await
            .expect_err("local query options are invalid remotely");
        let chat_error = endpoint
            .chat(CogentChatRequest {
                messages: vec![ChatMessage::new(ChatRole::User, "hello")],
                local: LocalTextOptions {
                    context_key: Some("ctx".to_string()),
                    ..LocalTextOptions::default()
                },
                ..CogentChatRequest::default()
            })
            .await
            .expect_err("local chat options are invalid remotely");
        let embed_error = endpoint
            .embed(CogentEmbedRequest {
                input: "abc".to_string(),
                local: LocalEmbedOptions {
                    normalize: Some(true),
                    ..LocalEmbedOptions::default()
                },
                ..CogentEmbedRequest::default()
            })
            .await
            .expect_err("local embed options are invalid remotely");

        assert!(matches!(query_error, CogentError::InvalidRequest(_)));
        assert!(matches!(chat_error, CogentError::InvalidRequest(_)));
        assert!(matches!(embed_error, CogentError::InvalidRequest(_)));
    });
}

#[test]
fn remote_response_future_reports_join_failures() {
    let executor = RemoteExecutor::new().expect("remote executor");
    let join = executor
        .spawn(async { futures::future::pending::<CogentResult<CogentTextResponse>>().await });
    join.abort();
    let error = sync_block_on(RemoteResponseFuture::new(join, executor)).expect_err("join error");

    assert!(matches!(
        error,
        CogentError::Internal(message) if message.contains("remote task failed")
    ));
}

#[test]
fn dropping_remote_response_future_aborts_task() {
    let executor = RemoteExecutor::new().expect("remote executor");
    let join = executor
        .spawn(async { futures::future::pending::<CogentResult<CogentTextResponse>>().await });
    let future = RemoteResponseFuture::new(join, executor);

    drop(future);
}
