// //! Tests the `remote_endpoint` module in `cogentlm-client`.
// //!
// //! Covers remote request mapping, streaming, provider error propagation, and
// //! task lifecycle behavior through fake provider backends without network I/O.

// use std::sync::{Arc, Mutex};

// use async_trait::async_trait;
// use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenBatch, TokenEmissionStats};
// use cogentlm_providers::{
//     ProviderBackend, ProviderChatRequest, ProviderChatResponse, ProviderEmbedRequest,
//     ProviderEmbeddingOutput, ProviderEmbeddingResponse, ProviderError, ProviderErrorKind,
//     ProviderGenerateRequest, ProviderGenerateResponse, ProviderKind, ProviderModel,
//     ProviderResponse, ProviderResponseMetadata, ProviderStream, ProviderStreamEvent,
//     ProviderTextOutput, ProviderTransport, TokenUsage,
// };
// use futures::executor::block_on;
// use futures::stream;
// use futures::StreamExt;
// use serde_json::json;

// use super::*;
// use crate::{CogentTextOptions, LocalTextOptions};

// #[derive(Default)]
// struct FakeBackend {
//     calls: Mutex<Vec<&'static str>>,
//     generate_error: Mutex<Option<ProviderErrorKind>>,
//     chat_error: Mutex<Option<ProviderErrorKind>>,
//     embed_error: Mutex<Option<ProviderErrorKind>>,
//     stream_events: Mutex<Option<Vec<cogentlm_providers::ProviderResult<ProviderStreamEvent>>>>,
// }

// impl FakeBackend {
//     fn with_generate_error(kind: ProviderErrorKind) -> Self {
//         Self {
//             generate_error: Mutex::new(Some(kind)),
//             ..Self::default()
//         }
//     }

//     fn with_chat_error(kind: ProviderErrorKind) -> Self {
//         Self {
//             chat_error: Mutex::new(Some(kind)),
//             ..Self::default()
//         }
//     }

//     fn with_embed_error(kind: ProviderErrorKind) -> Self {
//         Self {
//             embed_error: Mutex::new(Some(kind)),
//             ..Self::default()
//         }
//     }

//     fn with_stream_events(
//         events: Vec<cogentlm_providers::ProviderResult<ProviderStreamEvent>>,
//     ) -> Self {
//         Self {
//             stream_events: Mutex::new(Some(events)),
//             ..Self::default()
//         }
//     }
// }

// #[async_trait]
// impl ProviderBackend for FakeBackend {
//     fn kind(&self) -> ProviderKind {
//         ProviderKind::Proxy
//     }

//     async fn list_models(&self) -> cogentlm_providers::ProviderResult<Vec<ProviderModel>> {
//         Err(unused_call("list_models"))
//     }

//     async fn get_model(&self, _model: &str) -> cogentlm_providers::ProviderResult<ProviderModel> {
//         Err(unused_call("get_model"))
//     }

//     async fn chat(
//         &self,
//         req: ProviderChatRequest,
//     ) -> cogentlm_providers::ProviderResult<ProviderChatResponse> {
//         self.calls.lock().expect("calls").push("chat");
//         if let Some(kind) = self.chat_error.lock().expect("chat error").take() {
//             return Err(provider_error(kind, "chat failed"));
//         }
//         Ok(text_response(
//             &req.model,
//             req.messages
//                 .first()
//                 .map(|message| message.content.as_str())
//                 .unwrap_or_default(),
//         ))
//     }

//     async fn generate(
//         &self,
//         req: ProviderGenerateRequest,
//     ) -> cogentlm_providers::ProviderResult<ProviderGenerateResponse> {
//         self.calls.lock().expect("calls").push("generate");
//         if let Some(kind) = self.generate_error.lock().expect("generate error").take() {
//             return Err(provider_error(kind, "generate failed"));
//         }
//         Ok(text_response(&req.model, &req.prompt))
//     }

//     async fn embed(
//         &self,
//         req: ProviderEmbedRequest,
//     ) -> cogentlm_providers::ProviderResult<ProviderEmbeddingResponse> {
//         self.calls.lock().expect("calls").push("embed");
//         if let Some(kind) = self.embed_error.lock().expect("embed error").take() {
//             return Err(provider_error(kind, "embed failed"));
//         }
//         Ok(ProviderResponse {
//             result: ProviderEmbeddingOutput {
//                 values: vec![1.0, 2.0, 3.0],
//             },
//             usage: Some(TokenUsage {
//                 input_tokens: Some(req.input.len() as u32),
//                 output_tokens: None,
//                 total_tokens: Some(req.input.len() as u32),
//             }),
//             metadata: metadata(&req.model),
//         })
//     }

//     async fn stream_chat(
//         &self,
//         req: ProviderChatRequest,
//     ) -> cogentlm_providers::ProviderResult<ProviderStream<ProviderStreamEvent>> {
//         self.calls.lock().expect("calls").push("stream_chat");
//         let events = self
//             .stream_events
//             .lock()
//             .expect("stream events")
//             .take()
//             .unwrap_or_else(|| {
//                 vec![
//                     Ok(ProviderStreamEvent::TokenBatch(token_batch("a"))),
//                     Ok(ProviderStreamEvent::TokenBatch(token_batch("b"))),
//                     Ok(ProviderStreamEvent::Usage {
//                         usage: TokenUsage {
//                             input_tokens: Some(2),
//                             output_tokens: Some(2),
//                             total_tokens: Some(4),
//                         },
//                     }),
//                     Ok(ProviderStreamEvent::Finished {
//                         finish_reason: FinishReason::Length,
//                     }),
//                 ]
//             });
//         assert_eq!(req.model, "remote-model");
//         Ok(Box::pin(stream::iter(events)))
//     }
// }

// fn unused_call(name: &'static str) -> ProviderError {
//     provider_error(
//         ProviderErrorKind::UnsupportedFeature,
//         format!("{name} is not used by this test"),
//     )
// }

// fn provider_error(kind: ProviderErrorKind, message: impl Into<String>) -> ProviderError {
//     ProviderError::new(kind, ProviderKind::Proxy, message)
// }

// fn metadata(model: &str) -> ProviderResponseMetadata {
//     ProviderResponseMetadata {
//         provider: ProviderKind::Proxy,
//         model: model.to_string(),
//         request_id: Some("req-1".to_string()),
//         response_id: Some("resp-1".to_string()),
//         finish_reason_raw: None,
//         raw: json!({}),
//     }
// }

// fn text_response(model: &str, text: &str) -> ProviderGenerateResponse {
//     ProviderResponse {
//         result: ProviderTextOutput {
//             text: format!("echo:{text}"),
//             finish_reason: FinishReason::Stop,
//         },
//         usage: Some(TokenUsage {
//             input_tokens: Some(1),
//             output_tokens: Some(1),
//             total_tokens: Some(2),
//         }),
//         metadata: metadata(model),
//     }
// }

// fn token_batch(text: &str) -> TokenBatch {
//     TokenBatch {
//         request_id: "req-1".to_string(),
//         stream_id: 7,
//         sequence_start: 0,
//         text: text.to_string(),
//         frame_count: 1,
//         byte_count: text.len() as u32,
//         stats: TokenEmissionStats {
//             frames_sent: 1,
//             bytes_sent: text.len() as u64,
//             batches_sent: 1,
//         },
//     }
// }

// fn endpoint(backend: Arc<FakeBackend>) -> RemoteEndpoint {
//     RemoteEndpoint::new(
//         EndpointRef::Remote {
//             id: "remote".to_string(),
//         },
//         "remote-model".to_string(),
//         EndpointCapabilities::unknown(),
//         ProviderTransport::from_backend(backend),
//         RemoteExecutor::new().expect("remote executor"),
//     )
// }

// #[test]
// fn remote_generation_options_preserve_common_text_fields() {
//     let options = remote_generation_options(CogentTextOptions {
//         max_tokens: Some(7),
//         temperature: Some(0.25),
//         top_p: Some(0.9),
//         stop: vec!["stop".to_string()],
//     });

//     assert_eq!(options.max_tokens, Some(7));
//     assert_eq!(options.temperature, Some(0.25));
//     assert_eq!(options.top_p, Some(0.9));
//     assert_eq!(options.stop, vec!["stop"]);
// }

// #[test]
// fn capabilities_returns_configured_capability_snapshot() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(backend);

//     assert_eq!(endpoint.capabilities(), &EndpointCapabilities::unknown());
// }

// #[test]
// fn query_maps_provider_response_to_client_response() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(Arc::clone(&backend));
//     let response = block_on(endpoint.query(CogentQueryRequest {
//         prompt: "hello".to_string(),
//         options: CogentTextOptions {
//             max_tokens: Some(3),
//             ..CogentTextOptions::default()
//         },
//         ..CogentQueryRequest::default()
//     }))
//     .expect("query response");

//     assert_eq!(response.text, "echo:hello");
//     assert_eq!(response.finish_reason, FinishReason::Stop);
//     assert_eq!(response.endpoint, *endpoint.endpoint());
//     assert_eq!(
//         backend.calls.lock().expect("calls").as_slice(),
//         &["generate"]
//     );
// }

// #[test]
// fn query_rejects_token_emission_before_transport_call() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(Arc::clone(&backend));
//     let error = block_on(endpoint.query(CogentQueryRequest {
//         emit_tokens: true,
//         ..CogentQueryRequest::default()
//     }))
//     .expect_err("query token emission is unsupported");

//     assert!(matches!(
//         error,
//         CogentError::UnsupportedOperation {
//             operation: "query",
//             ..
//         }
//     ));
//     assert!(backend.calls.lock().expect("calls").is_empty());
// }

// #[test]
// fn query_rejects_local_options_before_transport_call() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(Arc::clone(&backend));
//     let error = block_on(endpoint.query(CogentQueryRequest {
//         local: LocalTextOptions {
//             grammar: Some("root ::= \"ok\"".to_string()),
//             ..LocalTextOptions::default()
//         },
//         ..CogentQueryRequest::default()
//     }))
//     .expect_err("local options are invalid for remote query");

//     assert!(matches!(error, CogentError::InvalidRequest(_)));
//     assert!(backend.calls.lock().expect("calls").is_empty());
// }

// #[test]
// fn chat_without_token_emission_maps_provider_response() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(Arc::clone(&backend));
//     let run = endpoint.chat(CogentChatRequest {
//         messages: vec![ChatMessage::new(ChatRole::User, "hello")],
//         ..CogentChatRequest::default()
//     });
//     let (mut tokens, response) = run.into_parts();
//     let response = block_on(response).expect("chat response");

//     assert_eq!(response.text, "echo:hello");
//     assert_eq!(response.finish_reason, FinishReason::Stop);
//     assert_eq!(response.endpoint, *endpoint.endpoint());
//     assert_eq!(response.usage.expect("usage").total_tokens, Some(2));
//     assert!(block_on(tokens.next()).is_none());
//     assert_eq!(backend.calls.lock().expect("calls").as_slice(), &["chat"]);
// }

// #[test]
// fn chat_stream_forwards_token_batches_and_final_response() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(Arc::clone(&backend));
//     let run = endpoint.chat(CogentChatRequest {
//         messages: vec![ChatMessage::new(ChatRole::User, "hello")],
//         emit_tokens: true,
//         ..CogentChatRequest::default()
//     });
//     let (tokens, response) = run.into_parts();
//     let (response, tokens) = block_on(async {
//         let response = response.await.expect("chat response");
//         let tokens = tokens.collect::<Vec<_>>().await;
//         (response, tokens)
//     });

//     assert_eq!(response.text, "ab");
//     assert_eq!(response.finish_reason, FinishReason::Length);
//     assert_eq!(response.usage.expect("usage").total_tokens, Some(4));
//     assert_eq!(
//         tokens
//             .iter()
//             .map(|batch| batch.text.as_str())
//             .collect::<Vec<_>>(),
//         vec!["a", "b"]
//     );
//     assert_eq!(
//         backend.calls.lock().expect("calls").as_slice(),
//         &["stream_chat"]
//     );
// }

// #[test]
// fn remote_stream_defaults_to_stop_without_usage_or_finished_event() {
//     let backend = Arc::new(FakeBackend::with_stream_events(vec![Ok(
//         ProviderStreamEvent::TokenBatch(token_batch("only")),
//     )]));
//     let endpoint = endpoint(Arc::clone(&backend));
//     let response = block_on(endpoint.chat(CogentChatRequest {
//         messages: vec![ChatMessage::new(ChatRole::User, "hello")],
//         emit_tokens: true,
//         ..CogentChatRequest::default()
//     }))
//     .expect("chat response");

//     assert_eq!(response.text, "only");
//     assert_eq!(response.finish_reason, FinishReason::Stop);
//     assert!(response.usage.is_none());
//     assert_eq!(
//         backend.calls.lock().expect("calls").as_slice(),
//         &["stream_chat"]
//     );
// }

// #[test]
// fn remote_stream_event_errors_surface_as_remote_errors() {
//     let backend = Arc::new(FakeBackend::with_stream_events(vec![
//         Ok(ProviderStreamEvent::TokenBatch(token_batch("a"))),
//         Err(provider_error(ProviderErrorKind::Timeout, "stream timeout")),
//     ]));
//     let endpoint = endpoint(Arc::clone(&backend));
//     let error = block_on(endpoint.chat(CogentChatRequest {
//         messages: vec![ChatMessage::new(ChatRole::User, "hello")],
//         emit_tokens: true,
//         ..CogentChatRequest::default()
//     }))
//     .expect_err("stream error");

//     assert!(matches!(
//         error,
//         CogentError::Remote(crate::RemoteError {
//             kind: crate::RemoteErrorKind::Timeout,
//             ..
//         })
//     ));
//     assert_eq!(
//         backend.calls.lock().expect("calls").as_slice(),
//         &["stream_chat"]
//     );
// }

// #[test]
// fn chat_rejects_local_options_before_transport_call() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(Arc::clone(&backend));
//     let error = block_on(endpoint.chat(CogentChatRequest {
//         local: LocalTextOptions {
//             context_key: Some("ctx".to_string()),
//             ..LocalTextOptions::default()
//         },
//         ..CogentChatRequest::default()
//     }))
//     .expect_err("local options are invalid for remote chat");

//     assert!(matches!(error, CogentError::InvalidRequest(_)));
//     assert!(backend.calls.lock().expect("calls").is_empty());
// }

// #[test]
// fn provider_errors_are_mapped_for_query_chat_and_embed() {
//     let query_backend = Arc::new(FakeBackend::with_generate_error(
//         ProviderErrorKind::RateLimited,
//     ));
//     let query_endpoint = endpoint(Arc::clone(&query_backend));
//     let query_error = block_on(query_endpoint.query(CogentQueryRequest::default()))
//         .expect_err("query provider error");
//     assert!(matches!(
//         query_error,
//         CogentError::Remote(crate::RemoteError {
//             kind: crate::RemoteErrorKind::RateLimited,
//             ..
//         })
//     ));

//     let chat_backend = Arc::new(FakeBackend::with_chat_error(ProviderErrorKind::Overloaded));
//     let chat_endpoint = endpoint(Arc::clone(&chat_backend));
//     let chat_error = block_on(chat_endpoint.chat(CogentChatRequest::default()))
//         .expect_err("chat provider error");
//     assert!(matches!(
//         chat_error,
//         CogentError::Remote(crate::RemoteError {
//             kind: crate::RemoteErrorKind::Overloaded,
//             ..
//         })
//     ));

//     let embed_backend = Arc::new(FakeBackend::with_embed_error(ProviderErrorKind::Transport));
//     let embed_endpoint = endpoint(Arc::clone(&embed_backend));
//     let embed_error = block_on(embed_endpoint.embed(CogentEmbedRequest::default()))
//         .expect_err("embed provider error");
//     assert!(matches!(
//         embed_error,
//         CogentError::Remote(crate::RemoteError {
//             kind: crate::RemoteErrorKind::Transport,
//             ..
//         })
//     ));
// }

// #[test]
// fn embed_rejects_local_options_before_transport_call() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(Arc::clone(&backend));
//     let error = block_on(endpoint.embed(CogentEmbedRequest {
//         local: crate::LocalEmbedOptions {
//             normalize: Some(true),
//             ..crate::LocalEmbedOptions::default()
//         },
//         ..CogentEmbedRequest::default()
//     }))
//     .expect_err("local options are invalid for remote embed");

//     assert!(matches!(error, CogentError::InvalidRequest(_)));
//     assert!(backend.calls.lock().expect("calls").is_empty());
// }

// #[test]
// fn embed_maps_provider_response_to_client_response() {
//     let backend = Arc::new(FakeBackend::default());
//     let endpoint = endpoint(Arc::clone(&backend));
//     let response = block_on(endpoint.embed(CogentEmbedRequest {
//         input: "abc".to_string(),
//         ..CogentEmbedRequest::default()
//     }))
//     .expect("embed response");

//     assert_eq!(response.values, vec![1.0, 2.0, 3.0]);
//     assert_eq!(response.endpoint, *endpoint.endpoint());
//     assert_eq!(response.usage.expect("usage").input_tokens, Some(3));
//     assert_eq!(backend.calls.lock().expect("calls").as_slice(), &["embed"]);
// }

// #[test]
// fn remote_response_future_reports_join_failures() {
//     let executor = RemoteExecutor::new().expect("remote executor");
//     let join = executor
//         .spawn(async { futures::future::pending::<CogentResult<CogentTextResponse>>().await });
//     join.abort();
//     let error = block_on(RemoteResponseFuture::new(join, executor)).expect_err("join error");

//     assert!(matches!(
//         error,
//         CogentError::Internal(message) if message.contains("remote task failed")
//     ));
// }

// #[test]
// fn dropping_remote_response_future_aborts_task() {
//     let executor = RemoteExecutor::new().expect("remote executor");
//     let join = executor
//         .spawn(async { futures::future::pending::<CogentResult<CogentTextResponse>>().await });
//     let future = RemoteResponseFuture::new(join, executor);

//     drop(future);
// }
