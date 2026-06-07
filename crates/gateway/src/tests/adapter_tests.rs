//! Tests framework-neutral routing, metadata preservation, streaming,
//! cancellation, and request validation with model-free fake executors.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::task::{Context, Poll};

use cogentlm_client::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingResponse, CogentEmbeddingRun,
    CogentQueryRequest, CogentResponseMetadata, CogentTextResponse, CogentTextRun, EndpointRef,
};
use cogentlm_core::{FinishReason, TokenUsage};
use futures_util::StreamExt;

use super::*;
use crate::{
    ChatMessageBody, ChatRequestBody, EmbedRequestBody, GatewayCancellationReason,
    GatewayRequestContext, QueryRequestBody,
};

#[derive(Clone)]
struct FakeExecutor {
    cancelled: Arc<AtomicBool>,
    pending: bool,
}

impl GatewayExecutor for FakeExecutor {
    fn query(
        &self,
        context: &GatewayRequestContext,
        _request: CogentQueryRequest,
    ) -> CogentTextRun {
        text_run(context, self.cancelled.clone(), self.pending)
    }

    fn chat(&self, context: &GatewayRequestContext, _request: CogentChatRequest) -> CogentTextRun {
        text_run(context, self.cancelled.clone(), self.pending)
    }

    fn embed(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        CogentEmbeddingRun::from_response(Box::pin(async {
            Ok(CogentEmbeddingResponse {
                endpoint: endpoint(),
                values: vec![1.0, 2.0],
                usage: None,
                local_stats: None,
                pooling: None,
                normalized: None,
                metadata: CogentResponseMetadata::default(),
            })
        }))
    }
}

fn text_run(
    _context: &GatewayRequestContext,
    cancelled: Arc<AtomicBool>,
    pending: bool,
) -> CogentTextRun {
    if pending {
        return CogentTextRun::from_response(Box::pin(PendingTextFuture { dropped: cancelled }));
    }
    CogentTextRun::from_response(Box::pin(FakeTextFuture {
        response: Some(CogentTextResponse {
            endpoint: endpoint(),
            text: "ok".to_string(),
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage {
                input_tokens: Some(1),
                output_tokens: Some(1),
                total_tokens: Some(2),
            }),
            local_stats: None,
            metadata: CogentResponseMetadata {
                request_id: Some("request-1".to_string()),
                upstream_request_id: Some("upstream-1".to_string()),
                upstream_response_id: Some("response-1".to_string()),
            },
        }),
        dropped: cancelled,
    }))
}

struct PendingTextFuture {
    dropped: Arc<AtomicBool>,
}

impl std::future::Future for PendingTextFuture {
    type Output = cogentlm_client::CogentResult<CogentTextResponse>;

    fn poll(self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTextFuture {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

struct FakeTextFuture {
    response: Option<CogentTextResponse>,
    dropped: Arc<AtomicBool>,
}

impl std::future::Future for FakeTextFuture {
    type Output = cogentlm_client::CogentResult<CogentTextResponse>;

    fn poll(mut self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(Ok(self.response.take().expect("polled once")))
    }
}

impl Drop for FakeTextFuture {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

fn endpoint() -> EndpointRef {
    EndpointRef::Local {
        id: "local".to_string(),
    }
}

fn adapter(cancelled: Arc<AtomicBool>) -> GatewayAdapter {
    adapter_with_limits(cancelled, false, GatewayAliasLimits::default())
}

fn adapter_with_limits(
    cancelled: Arc<AtomicBool>,
    pending: bool,
    limits: GatewayAliasLimits,
) -> GatewayAdapter {
    GatewayAdapter::builder(FakeExecutor { cancelled, pending })
        .alias(GatewayAlias::new("model", endpoint(), OperationSet::all(), limits).expect("alias"))
        .expect("add alias")
        .build()
        .expect("adapter")
}

fn context() -> GatewayRequestContext {
    GatewayRequestContext::new("request-1", GatewayCaller::anonymous()).expect("context")
}

#[tokio::test]
async fn finite_operations_preserve_correlation_metadata() {
    let adapter = adapter(Arc::new(AtomicBool::new(false)));
    let output = adapter
        .query(
            &context(),
            QueryRequestBody {
                model: "model".to_string(),
                prompt: "hello".to_string(),
                max_tokens: None,
                temperature: None,
                top_p: None,
                stop: Vec::new(),
                stream: false,
                gateway_options: Default::default(),
            },
        )
        .await
        .expect("query");

    assert_eq!(output.text, "ok");
    assert_eq!(
        output.metadata.upstream_request_id.as_deref(),
        Some("upstream-1")
    );
}

#[tokio::test]
async fn query_chat_and_embed_route_to_the_alias_endpoint() {
    let adapter = adapter(Arc::new(AtomicBool::new(false)));
    let context = context();

    adapter
        .chat(
            &context,
            ChatRequestBody {
                model: "model".to_string(),
                messages: vec![ChatMessageBody {
                    role: cogentlm_core::ChatRole::User,
                    content: "hello".to_string(),
                }],
                max_tokens: None,
                temperature: None,
                top_p: None,
                stop: Vec::new(),
                stream: false,
                gateway_options: Default::default(),
            },
        )
        .await
        .expect("chat");
    let embedding = adapter
        .embed(
            &context,
            EmbedRequestBody {
                model: "model".to_string(),
                input: "hello".to_string(),
                gateway_options: Default::default(),
            },
        )
        .await
        .expect("embed");

    assert_eq!(embedding.values, vec![1.0, 2.0]);
}

#[tokio::test]
async fn streaming_response_emits_usage_and_finished() {
    let adapter = adapter(Arc::new(AtomicBool::new(false)));
    let request_context = context();
    let mut stream = adapter
        .stream_query(
            &request_context,
            QueryRequestBody {
                model: "model".to_string(),
                prompt: "hello".to_string(),
                max_tokens: None,
                temperature: None,
                top_p: None,
                stop: Vec::new(),
                stream: true,
                gateway_options: Default::default(),
            },
        )
        .expect("stream");

    assert!(matches!(
        stream.next().await.expect("usage").expect("usage event"),
        GatewayStreamEvent::Usage { .. }
    ));
    assert!(matches!(
        stream
            .next()
            .await
            .expect("finished")
            .expect("finished event"),
        GatewayStreamEvent::Finished { .. }
    ));
}

#[test]
fn request_ids_are_bounded_visible_ascii() {
    assert!(crate::validate_request_id("request-123").is_ok());
    assert!(crate::validate_request_id("request with spaces").is_err());
    assert!(crate::validate_request_id(&"x".repeat(crate::MAX_REQUEST_ID_BYTES + 1)).is_err());
}

#[tokio::test]
async fn context_cancellation_reaches_client_run() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let adapter = adapter_with_limits(
        cancelled.clone(),
        true,
        GatewayAliasLimits {
            global: GatewayRequestLimits {
                max_concurrent_requests: Some(1),
                ..GatewayRequestLimits::default()
            },
            ..GatewayAliasLimits::default()
        },
    );
    let request_context = context();
    let mut stream = adapter
        .stream_query(
            &request_context,
            QueryRequestBody {
                model: "model".to_string(),
                prompt: "hello".to_string(),
                max_tokens: None,
                temperature: None,
                top_p: None,
                stop: Vec::new(),
                stream: true,
                gateway_options: Default::default(),
            },
        )
        .expect("stream");

    request_context
        .cancellation
        .cancel(GatewayCancellationReason::ClientDisconnected);

    let error = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
        .await
        .expect("cancellation should wake the stream")
        .expect("terminal error")
        .expect_err("cancelled stream");
    assert_eq!(error.kind, GatewayErrorKind::ClientDisconnected);
    assert!(cancelled.load(Ordering::SeqCst));

    let replacement = adapter
        .stream_query(
            &context(),
            QueryRequestBody {
                model: "model".to_string(),
                prompt: "replacement".to_string(),
                max_tokens: None,
                temperature: None,
                top_p: None,
                stop: Vec::new(),
                stream: true,
                gateway_options: Default::default(),
            },
        )
        .expect("cancellation should release the concurrency permit");
    drop(replacement);
}
