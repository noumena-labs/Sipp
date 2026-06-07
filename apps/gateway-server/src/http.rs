use std::{
    collections::HashMap,
    convert::Infallible,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, Extension, Request, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER},
        HeaderMap, HeaderName, HeaderValue, Method, StatusCode,
    },
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use cogentlm_core::TokenUsage;
use cogentlm_gateway::{
    finish_reason, validate_request_id, ChatRequestBody, EmbedRequestBody, EmbeddingResponseBody,
    ErrorEnvelope, GatewayAdapter, GatewayCancellation, GatewayCancellationReason, GatewayError,
    GatewayErrorKind, GatewayRequestContext, GatewayResult, GatewayStream, GatewayStreamEvent,
    Operation, QueryRequestBody, TextResponseBody, UsageBody,
};
use futures_util::Stream;
use serde::Serialize;
use serde_json::json;
use tokio::sync::{Notify, RwLock};
use tower_http::cors::CorsLayer;

use crate::{
    config::LoadedToken,
    lifecycle::{LifecycleState, ServerLifecycle},
    metrics::GatewayMetrics,
};

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
const RETRY_AFTER_MS: HeaderName = HeaderName::from_static("retry-after-ms");
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Shared HTTP service for separate public and management listeners.
#[derive(Clone)]
pub struct GatewayHttpService {
    state: Arc<ServiceState>,
    max_request_bytes: usize,
    allowed_origins: Vec<HeaderValue>,
}

struct ServiceState {
    lifecycle: Arc<LifecycleState>,
    adapter: RwLock<Option<GatewayAdapter>>,
    tokens: Vec<LoadedToken>,
    active: Arc<ActiveRequests>,
    metrics: Arc<GatewayMetrics>,
}

impl GatewayHttpService {
    /// Create a service in the `Starting` state.
    pub fn new(
        tokens: Vec<LoadedToken>,
        max_request_bytes: usize,
        allowed_origins: &[String],
    ) -> GatewayResult<Self> {
        if tokens.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway server requires at least one bearer token",
            ));
        }
        if max_request_bytes == 0 {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "max_request_bytes must be greater than zero",
            ));
        }
        let origins = allowed_origins
            .iter()
            .map(|origin| {
                HeaderValue::from_str(origin).map_err(|_| {
                    GatewayError::new(GatewayErrorKind::InvalidRequest, "invalid CORS origin")
                })
            })
            .collect::<GatewayResult<Vec<_>>>()?;
        let metrics = Arc::new(GatewayMetrics::new());
        Ok(Self {
            state: Arc::new(ServiceState {
                lifecycle: Arc::new(LifecycleState::starting()),
                adapter: RwLock::new(None),
                tokens,
                active: Arc::new(ActiveRequests::new(metrics.clone())),
                metrics,
            }),
            max_request_bytes,
            allowed_origins: origins,
        })
    }

    /// Build the public inference router.
    pub fn public_router(&self) -> Router {
        let router = Router::new()
            .route("/v1/query", post(query))
            .route("/v1/chat", post(chat))
            .route("/v1/embed", post(embed))
            .with_state(self.state.clone())
            .layer(DefaultBodyLimit::max(self.max_request_bytes))
            .layer(middleware::from_fn(request_id_middleware));

        if self.allowed_origins.is_empty() {
            router
        } else {
            router.layer(
                CorsLayer::new()
                    .allow_methods([Method::POST, Method::OPTIONS])
                    .allow_headers([AUTHORIZATION, CONTENT_TYPE, X_REQUEST_ID.clone()])
                    .expose_headers([X_REQUEST_ID.clone(), RETRY_AFTER, RETRY_AFTER_MS.clone()])
                    .allow_origin(self.allowed_origins.clone()),
            )
        }
    }

    /// Build the management router.
    pub fn management_router(&self) -> Router {
        Router::new()
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .route("/metrics", get(metrics))
            .with_state(self.state.clone())
    }

    /// Publish a fully loaded adapter and become ready.
    pub async fn set_ready(&self, adapter: GatewayAdapter) {
        *self.state.adapter.write().await = Some(adapter);
        self.state.lifecycle.set(ServerLifecycle::Ready);
    }

    /// Mark endpoint loading as failed.
    pub fn set_failed(&self) {
        self.state.lifecycle.set(ServerLifecycle::Failed);
    }

    /// Begin draining and reject new public requests.
    pub fn begin_draining(&self) {
        self.state.lifecycle.set(ServerLifecycle::Draining);
    }

    /// Wait until all active inference requests have finished.
    pub async fn wait_for_idle(&self, timeout: Duration) -> bool {
        self.state.active.wait_for_idle(timeout).await
    }

    /// Cancel every active request with a shutdown reason.
    pub fn cancel_active_for_shutdown(&self) {
        self.state
            .active
            .cancel_all(GatewayCancellationReason::ServerShutdown);
    }

    /// Return the current lifecycle.
    pub fn lifecycle(&self) -> ServerLifecycle {
        self.state.lifecycle.get()
    }
}

#[derive(Clone)]
struct RequestId(String);

impl RequestId {
    fn as_str(&self) -> &str {
        &self.0
    }
}

struct GatewayHttpError(GatewayError);

impl From<GatewayError> for GatewayHttpError {
    fn from(error: GatewayError) -> Self {
        Self(error)
    }
}

impl IntoResponse for GatewayHttpError {
    fn into_response(self) -> Response {
        gateway_error_response(self.0)
    }
}

async fn healthz(State(state): State<Arc<ServiceState>>) -> impl IntoResponse {
    Json(ProbeBody {
        status: "ok",
        state: lifecycle_label(state.lifecycle.get()),
    })
}

async fn readyz(State(state): State<Arc<ServiceState>>) -> Response {
    let lifecycle = state.lifecycle.get();
    let body = Json(ProbeBody {
        status: if lifecycle == ServerLifecycle::Ready {
            "ready"
        } else {
            "not_ready"
        },
        state: lifecycle_label(lifecycle),
    });
    if lifecycle == ServerLifecycle::Ready {
        (StatusCode::OK, body).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
    }
}

async fn metrics(State(state): State<Arc<ServiceState>>) -> Response {
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.metrics.render(),
    )
        .into_response()
}

async fn query(
    State(state): State<Arc<ServiceState>>,
    Extension(request_id): Extension<RequestId>,
    headers: HeaderMap,
    body: Result<Json<QueryRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    ensure_ready(&state)?;
    let caller = authorize(&state.tokens, &headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let stream = body.stream;
    let alias = body.model.clone();
    let context = GatewayRequestContext::new(request_id.0.clone(), caller)?;
    let adapter = adapter(&state).await?;
    let guard = state
        .active
        .register(context.cancellation.clone(), Operation::Query, stream);

    if stream {
        let stream = match adapter.stream_query(&context, body) {
            Ok(stream) => stream,
            Err(error) => {
                log_error(
                    &request_id,
                    &alias,
                    Operation::Query,
                    &error,
                    guard.elapsed(),
                );
                guard.complete(false);
                return Err(error.into());
            }
        };
        return Ok(sse_response(
            stream,
            guard,
            request_id,
            alias,
            Operation::Query,
        ));
    }

    match adapter.query(&context, body).await {
        Ok(output) => {
            if let Some(usage) = output.usage {
                state.metrics.usage(usage);
            }
            log_success(
                &request_id,
                &alias,
                Operation::Query,
                output.metadata.upstream_request_id.as_deref(),
                output.metadata.upstream_response_id.as_deref(),
                guard.elapsed(),
            );
            guard.complete(true);
            Ok(Json(TextResponseBody {
                id: request_id.0,
                model: alias,
                text: output.text,
                finish_reason: finish_reason(output.finish_reason),
                usage: output.usage.and_then(usage_body),
            })
            .into_response())
        }
        Err(error) => {
            log_error(
                &request_id,
                &alias,
                Operation::Query,
                &error,
                guard.elapsed(),
            );
            guard.complete(false);
            Err(error.into())
        }
    }
}

async fn chat(
    State(state): State<Arc<ServiceState>>,
    Extension(request_id): Extension<RequestId>,
    headers: HeaderMap,
    body: Result<Json<ChatRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    ensure_ready(&state)?;
    let caller = authorize(&state.tokens, &headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let stream = body.stream;
    let alias = body.model.clone();
    let context = GatewayRequestContext::new(request_id.0.clone(), caller)?;
    let adapter = adapter(&state).await?;
    let guard = state
        .active
        .register(context.cancellation.clone(), Operation::Chat, stream);

    if stream {
        let stream = match adapter.stream_chat(&context, body) {
            Ok(stream) => stream,
            Err(error) => {
                log_error(
                    &request_id,
                    &alias,
                    Operation::Chat,
                    &error,
                    guard.elapsed(),
                );
                guard.complete(false);
                return Err(error.into());
            }
        };
        return Ok(sse_response(
            stream,
            guard,
            request_id,
            alias,
            Operation::Chat,
        ));
    }

    match adapter.chat(&context, body).await {
        Ok(output) => {
            if let Some(usage) = output.usage {
                state.metrics.usage(usage);
            }
            log_success(
                &request_id,
                &alias,
                Operation::Chat,
                output.metadata.upstream_request_id.as_deref(),
                output.metadata.upstream_response_id.as_deref(),
                guard.elapsed(),
            );
            guard.complete(true);
            Ok(Json(TextResponseBody {
                id: request_id.0,
                model: alias,
                text: output.text,
                finish_reason: finish_reason(output.finish_reason),
                usage: output.usage.and_then(usage_body),
            })
            .into_response())
        }
        Err(error) => {
            log_error(
                &request_id,
                &alias,
                Operation::Chat,
                &error,
                guard.elapsed(),
            );
            guard.complete(false);
            Err(error.into())
        }
    }
}

async fn embed(
    State(state): State<Arc<ServiceState>>,
    Extension(request_id): Extension<RequestId>,
    headers: HeaderMap,
    body: Result<Json<EmbedRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    ensure_ready(&state)?;
    let caller = authorize(&state.tokens, &headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let alias = body.model.clone();
    let context = GatewayRequestContext::new(request_id.0.clone(), caller)?;
    let adapter = adapter(&state).await?;
    let guard = state
        .active
        .register(context.cancellation.clone(), Operation::Embed, false);

    match adapter.embed(&context, body).await {
        Ok(output) => {
            if let Some(usage) = output.usage {
                state.metrics.usage(usage);
            }
            log_success(
                &request_id,
                &alias,
                Operation::Embed,
                output.metadata.upstream_request_id.as_deref(),
                output.metadata.upstream_response_id.as_deref(),
                guard.elapsed(),
            );
            guard.complete(true);
            Ok(Json(EmbeddingResponseBody {
                id: request_id.0,
                model: alias,
                embedding: output.values,
                usage: output.usage.and_then(usage_body),
            })
            .into_response())
        }
        Err(error) => {
            log_error(
                &request_id,
                &alias,
                Operation::Embed,
                &error,
                guard.elapsed(),
            );
            guard.complete(false);
            Err(error.into())
        }
    }
}

async fn adapter(state: &ServiceState) -> GatewayResult<GatewayAdapter> {
    state.adapter.read().await.clone().ok_or_else(restarting)
}

fn ensure_ready(state: &ServiceState) -> GatewayResult<()> {
    if state.lifecycle.is_ready() {
        Ok(())
    } else {
        Err(restarting())
    }
}

fn restarting() -> GatewayError {
    GatewayError::new(
        GatewayErrorKind::ServerRestarting,
        "gateway is starting or restarting",
    )
}

fn authorize(
    tokens: &[LoadedToken],
    headers: &HeaderMap,
) -> GatewayResult<cogentlm_gateway::GatewayCaller> {
    let header = headers.get(AUTHORIZATION).ok_or_else(|| {
        GatewayError::new(GatewayErrorKind::Authentication, "missing bearer token")
    })?;
    let value = header.to_str().map_err(|_| {
        GatewayError::new(
            GatewayErrorKind::Authentication,
            "invalid authorization header",
        )
    })?;
    let secret = value.strip_prefix("Bearer ").ok_or_else(|| {
        GatewayError::new(
            GatewayErrorKind::Authentication,
            "authorization header must use bearer auth",
        )
    })?;
    let mut matched = None;
    for token in tokens {
        if constant_time_eq(secret.as_bytes(), token.secret.as_bytes()) {
            matched = Some(token.caller.clone());
        }
    }
    matched
        .ok_or_else(|| GatewayError::new(GatewayErrorKind::Authentication, "invalid bearer token"))
}

pub(crate) fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let max_len = left.len().max(right.len());
    let mut difference = left.len() ^ right.len();
    for index in 0..max_len {
        let left_byte = left.get(index).copied().unwrap_or_default();
        let right_byte = right.get(index).copied().unwrap_or_default();
        difference |= usize::from(left_byte ^ right_byte);
    }
    difference == 0
}

async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    let request_id = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|value| value.to_str().ok())
        .filter(|value| validate_request_id(value).is_ok())
        .map(str::to_owned)
        .unwrap_or_else(generated_request_id);
    request
        .extensions_mut()
        .insert(RequestId(request_id.clone()));
    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(X_REQUEST_ID.clone(), value);
    }
    response
}

fn generated_request_id() -> String {
    let id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    format!("gw_{id:016x}")
}

fn gateway_error_response(error: GatewayError) -> Response {
    let mut headers = HeaderMap::new();
    if let Some(retry_after) = error.retry_after {
        insert_header(&mut headers, RETRY_AFTER, retry_after.as_secs().to_string());
        insert_header(
            &mut headers,
            RETRY_AFTER_MS.clone(),
            retry_after.as_millis().to_string(),
        );
    }
    let status = StatusCode::from_u16(error.kind.http_status_code())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, headers, Json(ErrorEnvelope::from(&error))).into_response()
}

fn json_rejection_error(rejection: JsonRejection) -> GatewayHttpError {
    let kind = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        GatewayErrorKind::RequestTooLarge
    } else {
        GatewayErrorKind::InvalidRequest
    };
    GatewayError::new(kind, "invalid JSON request body").into()
}

fn insert_header(headers: &mut HeaderMap, name: HeaderName, value: String) {
    if let Ok(value) = HeaderValue::from_str(&value) {
        headers.insert(name, value);
    }
}

fn sse_response(
    stream: GatewayStream,
    guard: RequestGuard,
    request_id: RequestId,
    alias: String,
    operation: Operation,
) -> Response {
    Sse::new(HttpSseStream {
        inner: stream,
        guard: Some(guard),
        request_id,
        alias,
        operation,
        terminal: false,
    })
    .keep_alive(KeepAlive::default())
    .into_response()
}

struct HttpSseStream {
    inner: GatewayStream,
    guard: Option<RequestGuard>,
    request_id: RequestId,
    alias: String,
    operation: Operation,
    terminal: bool,
}

impl Stream for HttpSseStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(GatewayStreamEvent::TokenBatch(batch)))) => {
                Poll::Ready(Some(Ok(Event::default()
                    .event("token")
                    .json_data(json!({
                        "text": batch.text,
                        "sequence": batch.sequence_start,
                    }))
                    .unwrap_or_else(internal_sse_error))))
            }
            Poll::Ready(Some(Ok(GatewayStreamEvent::Usage { usage }))) => {
                if let Some(guard) = &self.guard {
                    guard.metrics.usage(usage);
                }
                Poll::Ready(Some(Ok(Event::default()
                    .event("usage")
                    .json_data(UsageBody::from(usage))
                    .unwrap_or_else(internal_sse_error))))
            }
            Poll::Ready(Some(Ok(GatewayStreamEvent::Finished {
                finish_reason: reason,
                metadata,
            }))) => {
                if let Some(guard) = self.guard.take() {
                    log_success(
                        &self.request_id,
                        &self.alias,
                        self.operation,
                        metadata.upstream_request_id.as_deref(),
                        metadata.upstream_response_id.as_deref(),
                        guard.elapsed(),
                    );
                    guard.complete(true);
                }
                self.terminal = true;
                Poll::Ready(Some(Ok(Event::default()
                    .event("done")
                    .json_data(json!({ "finish_reason": reason.as_str() }))
                    .unwrap_or_else(internal_sse_error))))
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(guard) = self.guard.take() {
                    log_error(
                        &self.request_id,
                        &self.alias,
                        self.operation,
                        &error,
                        guard.elapsed(),
                    );
                    guard.complete(false);
                }
                self.terminal = true;
                Poll::Ready(Some(Ok(Event::default()
                    .event("error")
                    .json_data(ErrorEnvelope::from(&error))
                    .unwrap_or_else(internal_sse_error))))
            }
            Poll::Ready(None) => {
                self.terminal = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn internal_sse_error(_: axum::Error) -> Event {
    Event::default()
        .event("error")
        .data(r#"{"error":{"code":"internal","message":"failed to encode SSE event"}}"#)
}

struct ActiveRequests {
    next_id: AtomicU64,
    entries: Mutex<HashMap<u64, GatewayCancellation>>,
    notify: Notify,
    metrics: Arc<GatewayMetrics>,
}

impl ActiveRequests {
    fn new(metrics: Arc<GatewayMetrics>) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            entries: Mutex::new(HashMap::new()),
            notify: Notify::new(),
            metrics,
        }
    }

    fn register(
        self: &Arc<Self>,
        cancellation: GatewayCancellation,
        operation: Operation,
        stream: bool,
    ) -> RequestGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut entries) = self.entries.lock() {
            entries.insert(id, cancellation.clone());
        }
        self.metrics.request_started();
        if stream {
            self.metrics.stream_delta(1);
        }
        RequestGuard {
            registry: self.clone(),
            id,
            cancellation,
            operation,
            stream,
            started: Instant::now(),
            metrics: self.metrics.clone(),
            completed: false,
        }
    }

    fn unregister(&self, id: u64) {
        let empty = self
            .entries
            .lock()
            .map(|mut entries| {
                entries.remove(&id);
                entries.is_empty()
            })
            .unwrap_or(false);
        if empty {
            self.notify.notify_waiters();
        }
    }

    fn cancel_all(&self, reason: GatewayCancellationReason) {
        let cancellations = self
            .entries
            .lock()
            .map(|entries| entries.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        for cancellation in cancellations {
            if cancellation.reason().is_none() {
                self.metrics.cancellation(reason);
                cancellation.cancel(reason);
            }
        }
    }

    async fn wait_for_idle(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self
                .entries
                .lock()
                .map(|entries| entries.is_empty())
                .unwrap_or(true)
            {
                return true;
            }
            if tokio::time::timeout_at(deadline, self.notify.notified())
                .await
                .is_err()
            {
                return false;
            }
        }
    }
}

struct RequestGuard {
    registry: Arc<ActiveRequests>,
    id: u64,
    cancellation: GatewayCancellation,
    operation: Operation,
    stream: bool,
    started: Instant,
    metrics: Arc<GatewayMetrics>,
    completed: bool,
}

impl RequestGuard {
    fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    fn complete(mut self, success: bool) {
        self.finish(success);
    }

    fn finish(&mut self, success: bool) {
        if self.completed {
            return;
        }
        self.completed = true;
        self.metrics
            .request_finished(self.operation, success, self.started.elapsed());
        if self.stream {
            self.metrics.stream_delta(-1);
        }
        self.registry.unregister(self.id);
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if self.cancellation.reason().is_none() {
            self.metrics
                .cancellation(GatewayCancellationReason::ClientDisconnected);
            self.cancellation
                .cancel(GatewayCancellationReason::ClientDisconnected);
        }
        self.finish(false);
    }
}

fn usage_body(usage: TokenUsage) -> Option<UsageBody> {
    if usage.input_tokens.is_none() && usage.output_tokens.is_none() && usage.total_tokens.is_none()
    {
        None
    } else {
        Some(usage.into())
    }
}

fn log_success(
    request_id: &RequestId,
    alias: &str,
    operation: Operation,
    upstream_request_id: Option<&str>,
    upstream_response_id: Option<&str>,
    elapsed: Duration,
) {
    tracing::info!(
        target: "cogentlm_gateway_server::request",
        request_id = request_id.as_str(),
        alias,
        operation = operation.as_str(),
        outcome = "ok",
        latency_ms = elapsed.as_secs_f64() * 1_000.0,
        upstream_request_id,
        upstream_response_id,
        "gateway inference request"
    );
}

fn log_error(
    request_id: &RequestId,
    alias: &str,
    operation: Operation,
    error: &GatewayError,
    elapsed: Duration,
) {
    tracing::warn!(
        target: "cogentlm_gateway_server::request",
        request_id = request_id.as_str(),
        alias,
        operation = operation.as_str(),
        outcome = "error",
        error_code = error.code(),
        latency_ms = elapsed.as_secs_f64() * 1_000.0,
        upstream_request_id = error.upstream_request_id.as_deref(),
        "gateway inference request"
    );
}

fn lifecycle_label(state: ServerLifecycle) -> &'static str {
    match state {
        ServerLifecycle::Starting => "starting",
        ServerLifecycle::Ready => "ready",
        ServerLifecycle::Draining => "draining",
        ServerLifecycle::Failed => "failed",
    }
}

#[derive(Serialize)]
struct ProbeBody {
    status: &'static str,
    state: &'static str,
}
