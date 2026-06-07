//! Exercises public and management HTTP behavior with model-free executors,
//! including readiness transitions, request IDs, metrics, shutdown, and
//! downstream disconnect cancellation.

use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

use axum::{
    body::{to_bytes, Body},
    http::{header::AUTHORIZATION, Request, StatusCode},
};
use cogentlm_client::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingResponse, CogentEmbeddingRun,
    CogentQueryRequest, CogentResponseMetadata, CogentTextResponse, CogentTextRun, EndpointRef,
};
use cogentlm_core::FinishReason;
use cogentlm_gateway::{
    GatewayAdapter, GatewayAlias, GatewayAliasLimits, GatewayCaller, GatewayExecutor,
    GatewayRequestContext, OperationSet,
};
use cogentlm_gateway_server::{
    config::LoadedToken, http::GatewayHttpService, lifecycle::ServerLifecycle,
};
use serde_json::Value;
use tower::ServiceExt;

struct FakeExecutor;

impl GatewayExecutor for FakeExecutor {
    fn query(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentQueryRequest,
    ) -> CogentTextRun {
        CogentTextRun::from_response(Box::pin(async {
            Ok(CogentTextResponse {
                endpoint: endpoint(),
                text: "hello".to_string(),
                finish_reason: FinishReason::Stop,
                usage: None,
                local_stats: None,
                metadata: CogentResponseMetadata::default(),
            })
        }))
    }

    fn chat(&self, context: &GatewayRequestContext, request: CogentChatRequest) -> CogentTextRun {
        self.query(
            context,
            CogentQueryRequest {
                endpoint: request.endpoint,
                ..CogentQueryRequest::default()
            },
        )
    }

    fn embed(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        CogentEmbeddingRun::from_response(Box::pin(async {
            Ok(CogentEmbeddingResponse {
                endpoint: endpoint(),
                values: vec![1.0],
                usage: None,
                local_stats: None,
                pooling: None,
                normalized: None,
                metadata: CogentResponseMetadata::default(),
            })
        }))
    }
}

struct PendingExecutor {
    dropped: Arc<AtomicBool>,
}

impl GatewayExecutor for PendingExecutor {
    fn query(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentQueryRequest,
    ) -> CogentTextRun {
        CogentTextRun::from_response(Box::pin(PendingTextFuture {
            dropped: self.dropped.clone(),
        }))
    }

    fn chat(&self, context: &GatewayRequestContext, request: CogentChatRequest) -> CogentTextRun {
        self.query(
            context,
            CogentQueryRequest {
                endpoint: request.endpoint,
                ..CogentQueryRequest::default()
            },
        )
    }

    fn embed(
        &self,
        _context: &GatewayRequestContext,
        _request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        unreachable!("pending executor is only used for text streams")
    }
}

struct PendingTextFuture {
    dropped: Arc<AtomicBool>,
}

impl Future for PendingTextFuture {
    type Output = cogentlm_client::CogentResult<CogentTextResponse>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTextFuture {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

fn endpoint() -> EndpointRef {
    EndpointRef::Local {
        id: "local".to_string(),
    }
}

fn adapter() -> GatewayAdapter {
    adapter_with(FakeExecutor)
}

fn adapter_with(executor: impl GatewayExecutor + 'static) -> GatewayAdapter {
    GatewayAdapter::builder(executor)
        .alias(
            GatewayAlias::new(
                "local",
                endpoint(),
                OperationSet::all(),
                GatewayAliasLimits::default(),
            )
            .expect("alias"),
        )
        .expect("register alias")
        .build()
        .expect("adapter")
}

fn service() -> GatewayHttpService {
    GatewayHttpService::new(
        vec![LoadedToken::new("secret", GatewayCaller::anonymous()).expect("token")],
        1 << 20,
        &[],
    )
    .expect("service")
}

#[tokio::test]
async fn readiness_tracks_starting_ready_and_draining() {
    let service = service();
    let management = service.management_router();

    let response = management
        .clone()
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    service.set_ready(adapter()).await;
    let response = management
        .clone()
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    service.begin_draining();
    assert_eq!(service.lifecycle(), ServerLifecycle::Draining);
    let response = management
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn public_requests_are_rejected_until_ready_and_after_draining() {
    let service = service();
    let public = service.public_router();
    let request = || {
        Request::builder()
            .method("POST")
            .uri("/v1/query")
            .header(AUTHORIZATION, "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"local","prompt":"hi"}"#))
            .unwrap()
    };

    let response = public.clone().oneshot(request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "server_restarting");

    service.set_ready(adapter()).await;
    let response = public.clone().oneshot(request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    service.begin_draining();
    let response = public.oneshot(request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn request_id_is_propagated_to_the_response() {
    let service = service();
    service.set_ready(adapter()).await;
    let response = service
        .public_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/query")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .header("x-request-id", "caller-request-1")
                .body(Body::from(r#"{"model":"local","prompt":"hi"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok()),
        Some("caller-request-1")
    );
}

#[tokio::test]
async fn metrics_have_only_bounded_labels() {
    let service = service();
    let response = service
        .management_router()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();

    assert!(!text.contains("request_id="));
    assert!(!text.contains("caller="));
    assert!(!text.contains("provider="));
}

#[tokio::test]
async fn forced_shutdown_emits_terminal_server_restarting_event() {
    let dropped = Arc::new(AtomicBool::new(false));
    let service = service();
    service
        .set_ready(adapter_with(PendingExecutor {
            dropped: dropped.clone(),
        }))
        .await;
    let response = service
        .public_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/query")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"local","prompt":"hi","stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    service.cancel_active_for_shutdown();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();

    assert!(text.contains("event: error"));
    assert!(text.contains("server_restarting"));
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn dropping_stream_body_propagates_client_disconnect() {
    let dropped = Arc::new(AtomicBool::new(false));
    let service = service();
    service
        .set_ready(adapter_with(PendingExecutor {
            dropped: dropped.clone(),
        }))
        .await;
    let response = service
        .public_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/query")
                .header(AUTHORIZATION, "Bearer secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"local","prompt":"hi","stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    drop(response);

    assert!(dropped.load(Ordering::SeqCst));
}
