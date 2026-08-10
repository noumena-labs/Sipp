//! Tests the `local_endpoint` module in `sipp`.
//!
//! Covers validation, runtime dispatch, response mapping, and local error
//! propagation through a fake local runtime rather than a loaded native model.

use std::sync::{Arc, Mutex};

use super::*;
use crate::client::{LocalEmbedOptions, LocalTextOptions, SippTextOptions};
use crate::core::{ChatMessage, ChatRole, FinishReason};
use crate::engine::{
    ChatRequest, EmbedRequest, EmbeddingResult, GenerationResult, PoolingType, QueryRequest,
    RequestStats,
};
use crate::runtime::SynthesizedAudio;
use futures::executor::block_on;
use futures::StreamExt;

#[derive(Default)]
struct FakeLocalRuntime {
    calls: Mutex<Vec<&'static str>>,
    listen_requests: Mutex<Vec<ListenRequest>>,
    speak_requests: Mutex<Vec<SpeakRequest>>,
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
    fn close(&self) -> EndpointCloseFuture<'_> {
        Box::pin(async move {
            self.calls.lock().expect("calls").push("close");
            Ok(())
        })
    }

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
            |message| Err(crate::error::Error::RuntimeCommand(message.to_string())),
        );
        LocalTextRun {
            tokens: None,
            response: Box::pin(async move { result }),
        }
    }

    fn listen(&self, request: ListenRequest) -> LocalTextRun {
        self.calls.lock().expect("calls").push("listen");
        self.listen_requests
            .lock()
            .expect("listen requests")
            .push(request);
        let result = self.text_error.map_or_else(
            || {
                Ok(GenerationResult {
                    id: "listen-id".to_string(),
                    text: "transcript".to_string(),
                    finish_reason: FinishReason::Stop,
                    stats: RequestStats {
                        input_tokens: 6,
                        output_tokens: 2,
                        ..RequestStats::default()
                    },
                })
            },
            |message| Err(crate::error::Error::RuntimeCommand(message.to_string())),
        );
        LocalTextRun {
            tokens: None,
            response: Box::pin(async move { result }),
        }
    }

    fn speak(&self, request: SpeakRequest) -> EngineAudioRun {
        self.calls.lock().expect("calls").push("speak");
        self.speak_requests
            .lock()
            .expect("speak requests")
            .push(request);
        EngineAudioRun::from_response(Box::pin(async {
            Ok(SynthesizedAudio {
                data: b"RIFF....WAVE".to_vec(),
                sample_count: 24_000,
                sample_rate_hz: 24_000,
            })
        }))
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
            |message| Err(crate::error::Error::RuntimeCommand(message.to_string())),
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
            |message| Err(crate::error::Error::RuntimeCommand(message.to_string())),
        );
        Box::pin(async move { result })
    }
}

fn endpoint(runtime: Arc<dyn LocalRuntime>) -> LocalEndpoint {
    LocalEndpoint::from_runtime(
        EndpointRef::from_id("local"),
        EndpointCapabilities {
            query: crate::core::CapabilitySupport::Supported,
            chat: crate::core::CapabilitySupport::Supported,
            embed: crate::core::CapabilitySupport::Supported,
            listen: crate::core::CapabilitySupport::Supported,
            speak: crate::core::CapabilitySupport::Unsupported,
        },
        runtime,
    )
}

#[test]
fn close_waits_for_local_runtime_shutdown() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());

    block_on(endpoint.close()).expect("close local endpoint");

    assert_eq!(runtime.calls(), vec!["close"]);
}

#[test]
fn query_validates_before_local_runtime_dispatch() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let error = block_on(endpoint.query_with_context(
        SippRequestContext::default(),
        SippQueryRequest {
            extra: serde_json::Map::from_iter([("seed".to_string(), serde_json::json!(1))]),
            ..SippQueryRequest::default()
        },
    ))
    .expect_err("endpoint options are invalid locally");

    assert!(matches!(error, SippError::InvalidRequest(_)));
    assert!(runtime.calls().is_empty());
}

#[test]
fn query_maps_local_response_and_closes_missing_token_stream() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let run = endpoint.query_with_context(
        SippRequestContext::default(),
        SippQueryRequest {
            prompt: "hello".to_string(),
            options: SippTextOptions {
                max_tokens: Some(7),
                ..SippTextOptions::default()
            },
            emit_tokens: true,
            ..SippQueryRequest::default()
        },
    );
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
    let response = block_on(endpoint.chat_with_context(
        SippRequestContext::default(),
        SippChatRequest {
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            local: LocalTextOptions {
                context_key: Some("ctx".to_string()),
                ..LocalTextOptions::default()
            },
            ..SippChatRequest::default()
        },
    ))
    .expect("chat response");

    assert_eq!(response.endpoint, *endpoint.endpoint());
    assert_eq!(response.text, "chat:hello");
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(9));
    assert_eq!(runtime.calls(), vec!["chat"]);
}

#[test]
fn listen_dispatches_once_and_maps_the_existing_text_response() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let context = SippRequestContext {
        request_id: Some("request-1".to_string()),
    };
    let run = endpoint.listen_with_context(
        context,
        SippListenRequest {
            endpoint: Some(EndpointRef::from_id("local")),
            audio: vec![1, 2, 3],
            language: Some("en".to_string()),
            max_tokens: Some(96),
        },
    );
    let (mut tokens, response) = run.into_parts();
    let response = block_on(response).expect("listen response");

    assert_eq!(response.endpoint, *endpoint.endpoint());
    assert_eq!(response.text, "transcript");
    assert_eq!(response.usage.expect("usage").total_tokens, Some(8));
    assert_eq!(response.metadata.request_id.as_deref(), Some("request-1"));
    assert!(block_on(tokens.next()).is_none());
    assert_eq!(runtime.calls(), vec!["listen"]);
    assert_eq!(
        *runtime.listen_requests.lock().expect("listen requests"),
        vec![ListenRequest {
            audio: vec![1, 2, 3],
            language: Some("en".to_string()),
            max_tokens: 96,
        }]
    );
}

#[test]
fn speak_dispatches_once_and_maps_native_wav_output() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let response = block_on(
        endpoint.speak_with_context(
            SippRequestContext {
                request_id: Some("request-2".to_string()),
            },
            SippSpeakRequest::new("hello")
                .language("en")
                .speaker([1, 2, 3])
                .max_duration_ms(2_000),
        ),
    )
    .expect("speak response");

    assert_eq!(response.endpoint, *endpoint.endpoint());
    assert_eq!(response.audio, b"RIFF....WAVE");
    assert_eq!(response.sample_rate_hz, 24_000);
    assert_eq!(response.channels, 1);
    assert_eq!(response.duration_ms, 1_000);
    assert_eq!(response.metadata.request_id.as_deref(), Some("request-2"));
    assert_eq!(runtime.calls(), vec!["speak"]);
    assert_eq!(
        *runtime.speak_requests.lock().expect("speak requests"),
        vec![SpeakRequest {
            text: "hello".to_string(),
            language: Some("en".to_string()),
            speaker_audio: Some(vec![1, 2, 3]),
            max_duration_ms: Some(2_000),
        }]
    );
}

#[test]
fn speak_preserves_absent_backend_hints_until_the_native_boundary() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());

    block_on(endpoint.speak_with_context(
        SippRequestContext::default(),
        SippSpeakRequest::new("hello"),
    ))
    .expect("speak response");

    assert_eq!(
        *runtime.speak_requests.lock().expect("speak requests"),
        vec![SpeakRequest {
            text: "hello".to_string(),
            language: None,
            speaker_audio: None,
            max_duration_ms: None,
        }]
    );
}

#[test]
fn embed_maps_local_response() {
    let runtime = Arc::new(FakeLocalRuntime::default());
    let endpoint = endpoint(runtime.clone());
    let response = block_on(endpoint.embed_with_context(
        SippRequestContext::default(),
        SippEmbedRequest {
            input: "abc".to_string(),
            local: LocalEmbedOptions {
                normalize: Some(false),
                ..LocalEmbedOptions::default()
            },
            ..SippEmbedRequest::default()
        },
    ))
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
    let error = block_on(endpoint.query_with_context(
        SippRequestContext::default(),
        SippQueryRequest {
            prompt: "hello".to_string(),
            ..SippQueryRequest::default()
        },
    ))
    .expect_err("local text error");

    assert!(matches!(
        error,
        SippError::Local(crate::error::Error::RuntimeCommand(message))
            if message == "text failed"
    ));
    assert_eq!(runtime.calls(), vec!["query"]);
}

#[test]
fn local_embed_runtime_errors_are_wrapped() {
    let runtime = FakeLocalRuntime::embed_error("embed failed");
    let endpoint = endpoint(runtime.clone());
    let error = block_on(endpoint.embed_with_context(
        SippRequestContext::default(),
        SippEmbedRequest {
            input: "abc".to_string(),
            ..SippEmbedRequest::default()
        },
    ))
    .expect_err("local embed error");

    assert!(matches!(
        error,
        SippError::Local(crate::error::Error::RuntimeCommand(message))
            if message == "embed failed"
    ));
    assert_eq!(runtime.calls(), vec!["embed"]);
}
