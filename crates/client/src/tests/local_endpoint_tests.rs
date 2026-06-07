//! Tests the `local_endpoint` module in `cogentlm-client`.
//!
//! Covers validation, runtime dispatch, response mapping, and local error
//! propagation through a fake local runtime rather than a loaded native model.

use std::sync::{Arc, Mutex};

use super::*;
use crate::{CogentTextOptions, LocalEmbedOptions, LocalTextOptions};
use cogentlm_core::{ChatMessage, ChatRole, FinishReason};
use cogentlm_engine::engine::{
    ChatRequest, EmbedRequest, EmbeddingResult, GenerationResult, PoolingType, QueryRequest,
    RequestStats,
};
use futures::executor::block_on;
use futures::StreamExt;

#[derive(Default)]
struct FakeLocalRuntime {
    calls: Mutex<Vec<&'static str>>,
    text_error: Option<&'static str>,
    embed_error: Option<&'static str>,
}

impl FakeLocalRuntime {
    fn text_error(message: &'static str) -> Arc<Self> {
        Arc::new(Self {
            text_error: Some(message),
            ..Self::default()
        })
    }

    fn embed_error(message: &'static str) -> Arc<Self> {
        Arc::new(Self {
            embed_error: Some(message),
            ..Self::default()
        })
    }

    fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().expect("calls").clone()
    }
}

impl LocalRuntime for FakeLocalRuntime {
    fn query(&self, request: QueryRequest) -> LocalTextRun {
        self.calls.lock().expect("calls").push("query");
        let result = self.text_error.map_or_else(
            || {
                Ok(GenerationResult {
                    id: "query-id".to_string(),
                    text: format!("query:{}", request.prompt),
                    finish_reason: FinishReason::Stop,
                    stats: RequestStats {
                        input_tokens: 2,
                        output_tokens: 3,
                        ..RequestStats::default()
                    },
                })
            },
            |message| Err(cogentlm_engine::Error::RuntimeCommand(message.to_string())),
        );
        LocalTextRun {
            tokens: None,
            response: Box::pin(async move { result }),
        }
    }

    fn chat(&self, request: ChatRequest) -> LocalTextRun {
        self.calls.lock().expect("calls").push("chat");
        let text = request
            .messages
            .first()
            .map(|message| message.content.clone())
            .unwrap_or_default();
        let result = self.text_error.map_or_else(
            || {
                Ok(GenerationResult {
                    id: "chat-id".to_string(),
                    text: format!("chat:{text}"),
                    finish_reason: FinishReason::Length,
                    stats: RequestStats {
                        input_tokens: 4,
                        output_tokens: 5,
                        ..RequestStats::default()
                    },
                })
            },
            |message| Err(cogentlm_engine::Error::RuntimeCommand(message.to_string())),
        );
        LocalTextRun {
            tokens: None,
            response: Box::pin(async move { result }),
        }
    }

    fn embed(&self, request: EmbedRequest) -> EngineEmbeddingResponseFuture {
        self.calls.lock().expect("calls").push("embed");
        let result = self.embed_error.map_or_else(
            || {
                Ok(EmbeddingResult {
                    id: "embed-id".to_string(),
                    values: vec![request.input.len() as f32, 1.0],
                    pooling: PoolingType::Mean,
                    normalized: request.options.normalize,
                    stats: RequestStats {
                        input_tokens: request.input.len() as i32,
                        output_tokens: 0,
                        ..RequestStats::default()
                    },
                })
            },
            |message| Err(cogentlm_engine::Error::RuntimeCommand(message.to_string())),
        );
        Box::pin(async move { result })
    }
}

fn endpoint(runtime: Arc<dyn LocalRuntime>) -> LocalEndpoint {
    LocalEndpoint::from_runtime(
        EndpointRef::Local {
            id: "local".to_string(),
        },
        EndpointCapabilities {
            query: cogentlm_core::CapabilitySupport::Supported,
            chat: cogentlm_core::CapabilitySupport::Supported,
            embed: cogentlm_core::CapabilitySupport::Supported,
        },
        runtime,
    )
}

#[test]
fn query_validates_before_local_runtime_dispatch() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let error = block_on(endpoint.query(CogentQueryRequest {
        endpoint_options: serde_json::Map::from_iter([("seed".to_string(), serde_json::json!(1))]),
        ..CogentQueryRequest::default()
    }))
    .expect_err("endpoint options are invalid locally");

    assert!(matches!(error, CogentError::InvalidRequest(_)));
    assert!(runtime.calls().is_empty());
}

#[test]
fn query_maps_local_response_and_closes_missing_token_stream() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let run = endpoint.query(CogentQueryRequest {
        prompt: "hello".to_string(),
        options: CogentTextOptions {
            max_tokens: Some(7),
            ..CogentTextOptions::default()
        },
        emit_tokens: true,
        ..CogentQueryRequest::default()
    });
    let (mut tokens, response) = run.into_parts();
    let response = block_on(response).expect("query response");

    assert_eq!(response.endpoint, *endpoint.endpoint());
    assert_eq!(response.text, "query:hello");
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(5));
    assert_eq!(response.local_stats.expect("stats").input_tokens, 2);
    assert!(block_on(tokens.next()).is_none());
    assert_eq!(runtime.calls(), vec!["query"]);
}

#[test]
fn chat_maps_local_response() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let response = block_on(endpoint.chat(CogentChatRequest {
        messages: vec![ChatMessage::new(ChatRole::User, "hello")],
        local: LocalTextOptions {
            context_key: Some("ctx".to_string()),
            ..LocalTextOptions::default()
        },
        ..CogentChatRequest::default()
    }))
    .expect("chat response");

    assert_eq!(response.endpoint, *endpoint.endpoint());
    assert_eq!(response.text, "chat:hello");
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(9));
    assert_eq!(runtime.calls(), vec!["chat"]);
}

#[test]
fn embed_maps_local_response() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let response = block_on(endpoint.embed(CogentEmbedRequest {
        input: "abc".to_string(),
        local: LocalEmbedOptions {
            normalize: Some(false),
            ..LocalEmbedOptions::default()
        },
        ..CogentEmbedRequest::default()
    }))
    .expect("embed response");

    assert_eq!(response.endpoint, *endpoint.endpoint());
    assert_eq!(response.values, vec![3.0, 1.0]);
    assert_eq!(response.usage.expect("usage").input_tokens, Some(3));
    assert_eq!(response.pooling, Some(PoolingType::Mean));
    assert_eq!(response.normalized, Some(false));
    assert_eq!(runtime.calls(), vec!["embed"]);
}

#[test]
fn local_text_runtime_errors_are_wrapped() {
    let runtime = FakeLocalRuntime::text_error("text failed");
    let endpoint = endpoint(runtime.clone());
    let error = block_on(endpoint.query(CogentQueryRequest {
        prompt: "hello".to_string(),
        ..CogentQueryRequest::default()
    }))
    .expect_err("local text error");

    assert!(matches!(
        error,
        CogentError::Local(cogentlm_engine::Error::RuntimeCommand(message))
            if message == "text failed"
    ));
    assert_eq!(runtime.calls(), vec!["query"]);
}

#[test]
fn local_embed_runtime_errors_are_wrapped() {
    let runtime = FakeLocalRuntime::embed_error("embed failed");
    let endpoint = endpoint(runtime.clone());
    let error = block_on(endpoint.embed(CogentEmbedRequest {
        input: "abc".to_string(),
        ..CogentEmbedRequest::default()
    }))
    .expect_err("local embed error");

    assert!(matches!(
        error,
        CogentError::Local(cogentlm_engine::Error::RuntimeCommand(message))
            if message == "embed failed"
    ));
    assert_eq!(runtime.calls(), vec!["embed"]);
}
