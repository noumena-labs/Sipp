use std::{
    collections::VecDeque,
    convert::Infallible,
    net::IpAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, Extension, Request, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER},
        HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri,
    },
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use cogentlm_core::TokenUsage;
use cogentlm_gateway::{
    constant_time_eq, finish_reason, BackendEmbeddingOutput, BackendTextOutput, ChatRequestBody,
    EmbedRequestBody, EmbeddingResponseBody, GatewayAccess, GatewayAdapter, GatewayAliasLimits,
    GatewayAliasSnapshot, GatewayCaller, GatewayError, GatewayErrorKind, GatewayResult,
    GatewayStream, GatewayStreamEvent, QueryRequestBody, TextResponseBody, UsageBody,
};
use futures_util::{Stream, StreamExt};
use serde::Serialize;
use serde_json::json;
use tower_http::cors::CorsLayer;

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
const RETRY_AFTER_MS: HeaderName = HeaderName::from_static("retry-after-ms");
static NEXT_GATEWAY_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Default maximum request body size accepted by the gateway server.
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 1 << 20;

/// Gateway-wide HTTP service limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayHttpLimits {
    /// Maximum JSON request body size in bytes.
    pub max_request_bytes: usize,
}

impl Default for GatewayHttpLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
        }
    }
}

/// Standalone gateway bearer token and access scope.
#[derive(Clone, PartialEq, Eq)]
pub struct GatewayToken {
    secret: String,
    access: GatewayAccess,
}

impl GatewayToken {
    /// Create a standalone server token.
    pub fn new(secret: impl Into<String>, access: GatewayAccess) -> GatewayResult<Self> {
        let secret = secret.into();
        cogentlm_gateway::validate_gateway_bearer_secret(&secret, "gateway token")?;
        Ok(Self { secret, access })
    }
}

/// Standalone HTTP service for a gateway adapter.
#[derive(Clone)]
pub struct GatewayHttpService {
    adapter: GatewayAdapter,
    tokens: Vec<GatewayToken>,
    admin_secret: String,
    allowed_origins: Vec<HeaderValue>,
    limits: GatewayHttpLimits,
    history: GatewayHistory,
    started: Instant,
}

impl GatewayHttpService {
    /// Create a standalone gateway HTTP service.
    pub fn new(
        adapter: GatewayAdapter,
        tokens: Vec<GatewayToken>,
        admin_secret: String,
        allowed_origins: Vec<String>,
        limits: GatewayHttpLimits,
        history_capacity: usize,
    ) -> GatewayResult<Self> {
        if tokens.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway server requires at least one bearer token",
            ));
        }
        cogentlm_gateway::validate_gateway_bearer_secret(&admin_secret, "gateway admin token")?;
        validate_service_limits(limits)?;
        let mut origins = Vec::with_capacity(allowed_origins.len());
        for origin in allowed_origins {
            origins.push(cors_origin_header(&origin)?);
        }
        Ok(Self {
            adapter,
            tokens,
            admin_secret,
            allowed_origins: origins,
            limits,
            history: GatewayHistory::new(history_capacity),
            started: Instant::now(),
        })
    }

    /// Build the Axum router.
    pub fn router(self) -> Router {
        let allowed_origins = self.allowed_origins.clone();
        let max_request_bytes = self.limits.max_request_bytes;
        let state = Arc::new(self);
        let router = Router::new()
            .route("/", get(dashboard))
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .route("/admin/api/status", get(status))
            .route("/admin/api/history", get(history))
            .route("/v1/query", post(query))
            .route("/v1/chat", post(chat))
            .route("/v1/embed", post(embed))
            .with_state(state)
            .layer(DefaultBodyLimit::max(max_request_bytes));

        let router = if allowed_origins.is_empty() {
            router
        } else {
            router.layer(
                CorsLayer::new()
                    .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                    .allow_headers([AUTHORIZATION, CONTENT_TYPE])
                    .expose_headers([X_REQUEST_ID.clone(), RETRY_AFTER, RETRY_AFTER_MS.clone()])
                    .allow_origin(allowed_origins),
            )
        };
        router.layer(middleware::from_fn(add_request_id))
    }
}

#[derive(Clone)]
struct GatewayHistory {
    inner: Arc<Mutex<GatewayHistoryInner>>,
}

struct GatewayHistoryInner {
    capacity: usize,
    entries: VecDeque<HistoryEntry>,
}

impl GatewayHistory {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(GatewayHistoryInner {
                capacity,
                entries: VecDeque::with_capacity(capacity),
            })),
        }
    }

    fn push(&self, entry: HistoryEntry) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if inner.entries.len() == inner.capacity {
            inner.entries.pop_front();
        }
        inner.entries.push_back(entry);
    }

    fn snapshot(&self) -> Vec<HistoryEntry> {
        let Ok(inner) = self.inner.lock() else {
            return Vec::new();
        };
        inner.entries.iter().cloned().collect()
    }
}

#[derive(Debug, Clone, Serialize)]
struct HistoryEntry {
    request_id: String,
    timestamp_ms: u128,
    alias: String,
    operation: &'static str,
    outcome: &'static str,
    status_code: u16,
    latency_ms: f64,
    stream: bool,
    finish_reason: Option<String>,
    usage: Option<UsageBody>,
    error_code: Option<String>,
}

#[derive(Clone)]
struct GatewayRequestId(String);

impl GatewayRequestId {
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

async fn dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn readyz() -> &'static str {
    "ready"
}

async fn status(
    State(state): State<Arc<GatewayHttpService>>,
    headers: HeaderMap,
) -> Result<Json<StatusBody>, GatewayHttpError> {
    state.authorize_admin(&headers)?;
    let snapshot = state.adapter.snapshot()?;
    Ok(Json(StatusBody {
        uptime_secs: state.started.elapsed().as_secs(),
        aliases: snapshot
            .aliases
            .into_iter()
            .map(AliasStatusBody::from)
            .collect(),
    }))
}

async fn history(
    State(state): State<Arc<GatewayHttpService>>,
    headers: HeaderMap,
) -> Result<Json<HistoryBody>, GatewayHttpError> {
    state.authorize_admin(&headers)?;
    Ok(Json(HistoryBody {
        entries: state.history.snapshot(),
    }))
}

async fn query(
    State(state): State<Arc<GatewayHttpService>>,
    Extension(request_id): Extension<GatewayRequestId>,
    headers: HeaderMap,
    body: Result<Json<QueryRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    let started = Instant::now();
    let caller = state.authorize_inference(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let model = body.model.clone();
    let stream = body.stream;
    let request = body.into_backend();
    if stream {
        let stream = match state.adapter.stream_query(&caller, &model, request).await {
            Ok(stream) => stream,
            Err(error) => {
                record_error(
                    &state,
                    &request_id,
                    &model,
                    OperationInfo::Query,
                    true,
                    started,
                    &error,
                );
                return Err(error.into());
            }
        };
        return Ok(sse(observe_stream(
            state.history.clone(),
            stream,
            request_id,
            model,
            OperationInfo::Query,
            started,
        ))
        .into_response());
    }
    match state.adapter.query(&caller, &model, request).await {
        Ok(output) => {
            record_text_success(
                &state,
                &request_id,
                &model,
                OperationInfo::Query,
                false,
                &output,
                started,
            );
            Ok(Json(text_response(&request_id, model, output)).into_response())
        }
        Err(error) => {
            record_error(
                &state,
                &request_id,
                &model,
                OperationInfo::Query,
                false,
                started,
                &error,
            );
            Err(error.into())
        }
    }
}

async fn chat(
    State(state): State<Arc<GatewayHttpService>>,
    Extension(request_id): Extension<GatewayRequestId>,
    headers: HeaderMap,
    body: Result<Json<ChatRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    let started = Instant::now();
    let caller = state.authorize_inference(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let model = body.model.clone();
    let stream = body.stream;
    let request = body.into_backend();
    if stream {
        let stream = match state.adapter.stream_chat(&caller, &model, request).await {
            Ok(stream) => stream,
            Err(error) => {
                record_error(
                    &state,
                    &request_id,
                    &model,
                    OperationInfo::Chat,
                    true,
                    started,
                    &error,
                );
                return Err(error.into());
            }
        };
        return Ok(sse(observe_stream(
            state.history.clone(),
            stream,
            request_id,
            model,
            OperationInfo::Chat,
            started,
        ))
        .into_response());
    }
    match state.adapter.chat(&caller, &model, request).await {
        Ok(output) => {
            record_text_success(
                &state,
                &request_id,
                &model,
                OperationInfo::Chat,
                false,
                &output,
                started,
            );
            Ok(Json(text_response(&request_id, model, output)).into_response())
        }
        Err(error) => {
            record_error(
                &state,
                &request_id,
                &model,
                OperationInfo::Chat,
                false,
                started,
                &error,
            );
            Err(error.into())
        }
    }
}

async fn embed(
    State(state): State<Arc<GatewayHttpService>>,
    Extension(request_id): Extension<GatewayRequestId>,
    headers: HeaderMap,
    body: Result<Json<EmbedRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    let started = Instant::now();
    let caller = state.authorize_inference(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let model = body.model.clone();
    let request = body.into_backend();
    match state.adapter.embed(&caller, &model, request).await {
        Ok(output) => {
            record_embedding_success(&state, &request_id, &model, &output, started);
            Ok(Json(embedding_response(&request_id, model, output)).into_response())
        }
        Err(error) => {
            record_error(
                &state,
                &request_id,
                &model,
                OperationInfo::Embed,
                false,
                started,
                &error,
            );
            Err(error.into())
        }
    }
}

impl GatewayHttpService {
    fn authorize_inference(&self, headers: &HeaderMap) -> GatewayResult<GatewayCaller> {
        let token = bearer_token(headers, "gateway inference")?;
        let mut matched_index = None;
        for (index, candidate) in self.tokens.iter().enumerate() {
            if constant_time_eq(token.as_bytes(), candidate.secret.as_bytes()) {
                matched_index = Some(index);
            }
        }
        matched_index
            .map(|index| GatewayCaller {
                id: Some(format!("standalone-token-{index}")),
                access: self.tokens[index].access.clone(),
            })
            .ok_or_else(|| {
                GatewayError::new(GatewayErrorKind::Authentication, "invalid bearer token")
            })
    }

    fn authorize_admin(&self, headers: &HeaderMap) -> GatewayResult<()> {
        let token = bearer_token(headers, "gateway admin")?;
        if constant_time_eq(token.as_bytes(), self.admin_secret.as_bytes()) {
            return Ok(());
        }
        Err(GatewayError::new(
            GatewayErrorKind::Authentication,
            "invalid admin bearer token",
        ))
    }
}

fn bearer_token<'a>(headers: &'a HeaderMap, label: &'static str) -> GatewayResult<&'a str> {
    let Some(header) = headers.get(AUTHORIZATION) else {
        return Err(GatewayError::new(
            GatewayErrorKind::Authentication,
            format!("missing {label} bearer token"),
        ));
    };
    let value = header.to_str().map_err(|_| {
        GatewayError::new(
            GatewayErrorKind::Authentication,
            "invalid authorization header",
        )
    })?;
    value.strip_prefix("Bearer ").ok_or_else(|| {
        GatewayError::new(
            GatewayErrorKind::Authentication,
            "authorization header must use bearer auth",
        )
    })
}

async fn add_request_id(mut request: Request, next: Next) -> Response {
    let request_id = next_request_id();
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let started = Instant::now();
    request
        .extensions_mut()
        .insert(GatewayRequestId(request_id.clone()));
    let mut response = next.run(request).await;
    let status = response.status().as_u16();
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(X_REQUEST_ID.clone(), value);
    }
    tracing::info!(
        target: "cogentlm_gateway_server::request",
        request_id = %request_id,
        method = %method,
        path = %path,
        status,
        latency_ms,
        "gateway request"
    );
    response
}

fn next_request_id() -> String {
    let id = NEXT_GATEWAY_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    format!("gw_{id:016x}")
}

fn text_response(
    request_id: &GatewayRequestId,
    model: String,
    output: BackendTextOutput,
) -> TextResponseBody {
    TextResponseBody {
        id: request_id.as_str().to_string(),
        model,
        text: output.text,
        finish_reason: finish_reason(output.finish_reason),
        usage: output.usage.and_then(usage_body),
    }
}

fn embedding_response(
    request_id: &GatewayRequestId,
    model: String,
    output: BackendEmbeddingOutput,
) -> EmbeddingResponseBody {
    EmbeddingResponseBody {
        id: request_id.as_str().to_string(),
        model,
        embedding: output.values,
        usage: output.usage.and_then(usage_body),
    }
}

fn usage_body(usage: TokenUsage) -> Option<UsageBody> {
    if usage.input_tokens.is_none() && usage.output_tokens.is_none() && usage.total_tokens.is_none()
    {
        None
    } else {
        Some(UsageBody::from(usage))
    }
}

fn sse(
    stream: impl Stream<Item = GatewayResult<GatewayStreamEvent>> + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Sse::new(stream.filter_map(|event| async move { sse_event(event).map(Ok) }))
        .keep_alive(KeepAlive::default())
}

fn observe_stream(
    history: GatewayHistory,
    stream: GatewayStream<GatewayStreamEvent>,
    request_id: GatewayRequestId,
    alias: String,
    operation: OperationInfo,
    started: Instant,
) -> GatewayStream<GatewayStreamEvent> {
    let mut usage = None;
    let mut recorded = false;
    Box::pin(stream.map(move |event| {
        match &event {
            Ok(GatewayStreamEvent::Usage { usage: current }) => {
                usage = Some(*current);
            }
            Ok(GatewayStreamEvent::Finished { finish_reason }) => {
                if !recorded {
                    history.push(HistoryEntry {
                        request_id: request_id.as_str().to_string(),
                        timestamp_ms: unix_ms(),
                        alias: alias.clone(),
                        operation: operation.as_str(),
                        outcome: "ok",
                        status_code: 200,
                        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
                        stream: true,
                        finish_reason: Some(finish_reason.as_str().to_string()),
                        usage: usage.and_then(usage_body),
                        error_code: None,
                    });
                    recorded = true;
                }
            }
            Err(error) => {
                if !recorded {
                    history.push(error_history_entry(
                        &request_id,
                        &alias,
                        operation,
                        true,
                        started,
                        error,
                    ));
                    recorded = true;
                }
            }
            Ok(GatewayStreamEvent::TokenBatch(_)) => {}
        }
        event
    }))
}

fn sse_event(event: GatewayResult<GatewayStreamEvent>) -> Option<Event> {
    match event {
        Ok(GatewayStreamEvent::TokenBatch(batch)) => Some(
            Event::default()
                .event("token")
                .json_data(json!({
                    "text": batch.text,
                    "sequence": batch.sequence_start,
                }))
                .unwrap_or_else(internal_sse_error),
        ),
        Ok(GatewayStreamEvent::Usage { usage }) => usage_body(usage).map(|usage| {
            Event::default()
                .event("usage")
                .json_data(usage)
                .unwrap_or_else(internal_sse_error)
        }),
        Ok(GatewayStreamEvent::Finished { finish_reason }) => Some(
            Event::default()
                .event("done")
                .json_data(json!({ "finish_reason": finish_reason.as_str() }))
                .unwrap_or_else(internal_sse_error),
        ),
        Err(error) => Some(
            Event::default()
                .event("error")
                .json_data(json!({
                    "error": {
                        "code": error.code(),
                        "message": error.message,
                    }
                }))
                .unwrap_or_else(internal_sse_error),
        ),
    }
}

fn internal_sse_error(_: axum::Error) -> Event {
    Event::default()
        .event("error")
        .data(r#"{"error":{"code":"internal","message":"failed to encode SSE event"}}"#)
}

fn record_text_success(
    state: &GatewayHttpService,
    request_id: &GatewayRequestId,
    alias: &str,
    operation: OperationInfo,
    stream: bool,
    output: &BackendTextOutput,
    started: Instant,
) {
    state.history.push(HistoryEntry {
        request_id: request_id.as_str().to_string(),
        timestamp_ms: unix_ms(),
        alias: alias.to_string(),
        operation: operation.as_str(),
        outcome: "ok",
        status_code: 200,
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
        stream,
        finish_reason: Some(output.finish_reason.as_str().to_string()),
        usage: output.usage.and_then(usage_body),
        error_code: None,
    });
}

fn record_embedding_success(
    state: &GatewayHttpService,
    request_id: &GatewayRequestId,
    alias: &str,
    output: &BackendEmbeddingOutput,
    started: Instant,
) {
    state.history.push(HistoryEntry {
        request_id: request_id.as_str().to_string(),
        timestamp_ms: unix_ms(),
        alias: alias.to_string(),
        operation: OperationInfo::Embed.as_str(),
        outcome: "ok",
        status_code: 200,
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
        stream: false,
        finish_reason: None,
        usage: output.usage.and_then(usage_body),
        error_code: None,
    });
}

fn record_error(
    state: &GatewayHttpService,
    request_id: &GatewayRequestId,
    alias: &str,
    operation: OperationInfo,
    stream: bool,
    started: Instant,
    error: &GatewayError,
) {
    state.history.push(error_history_entry(
        request_id, alias, operation, stream, started, error,
    ));
}

fn error_history_entry(
    request_id: &GatewayRequestId,
    alias: &str,
    operation: OperationInfo,
    stream: bool,
    started: Instant,
    error: &GatewayError,
) -> HistoryEntry {
    HistoryEntry {
        request_id: request_id.as_str().to_string(),
        timestamp_ms: unix_ms(),
        alias: alias.to_string(),
        operation: operation.as_str(),
        outcome: "error",
        status_code: error.kind.http_status_code(),
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
        stream,
        finish_reason: None,
        usage: None,
        error_code: Some(error.code().to_string()),
    }
}

#[derive(Debug, Clone, Copy)]
enum OperationInfo {
    Query,
    Chat,
    Embed,
}

impl OperationInfo {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Chat => "chat",
            Self::Embed => "embed",
        }
    }
}

fn json_rejection_error(rejection: JsonRejection) -> GatewayHttpError {
    if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return GatewayError::new(
            GatewayErrorKind::RequestTooLarge,
            "request body exceeds gateway limit",
        )
        .into();
    }
    GatewayError::new(
        GatewayErrorKind::InvalidRequest,
        "invalid JSON request body",
    )
    .into()
}

fn gateway_error_response(error: GatewayError) -> Response {
    let mut headers = HeaderMap::new();
    if let Some(retry_after) = error.retry_after {
        insert_header_if_valid(
            &mut headers,
            HeaderName::from_static("retry-after"),
            retry_after.as_secs().to_string(),
        );
        insert_header_if_valid(
            &mut headers,
            HeaderName::from_static("retry-after-ms"),
            retry_after.as_millis().to_string(),
        );
    }

    let status = StatusCode::from_u16(error.kind.http_status_code())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = Json(json!({
        "error": {
            "code": error.code(),
            "message": error.message,
        }
    }));
    (status, headers, body).into_response()
}

fn insert_header_if_valid(headers: &mut HeaderMap, name: HeaderName, value: impl AsRef<str>) {
    if let Ok(value) = HeaderValue::from_str(value.as_ref()) {
        headers.insert(name, value);
    }
}

fn cors_origin_header(origin: &str) -> GatewayResult<HeaderValue> {
    let trimmed = origin.trim();
    if trimmed.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "CORS origin must not be empty",
        ));
    }
    if trimmed != origin {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("invalid CORS origin {origin}: surrounding whitespace is not allowed"),
        ));
    }
    if trimmed == "*" || trimmed.eq_ignore_ascii_case("null") {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "CORS origin must be an exact application origin",
        ));
    }

    let uri = trimmed.parse::<Uri>().map_err(|error| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("invalid CORS origin {origin}: {error}"),
        )
    })?;
    let Some(scheme) = uri.scheme_str() else {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "CORS origin must be an absolute http(s) origin",
        ));
    };
    let Some(authority) = uri.authority() else {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "CORS origin must be an absolute http(s) origin",
        ));
    };
    if !matches!(scheme, "http" | "https") {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "CORS origin must be an absolute http(s) origin",
        ));
    }
    if authority.as_str().contains('@') {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "CORS origin must not include userinfo",
        ));
    }
    let expected_origin = format!("{scheme}://{authority}");
    if trimmed != expected_origin {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "CORS origin must not include a path, query, or fragment",
        ));
    }
    if scheme == "http" && !uri.host().is_some_and(is_loopback_host) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "CORS origin must use HTTPS unless it targets loopback",
        ));
    }

    HeaderValue::from_str(trimmed).map_err(|error| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("invalid CORS origin {origin}: {error}"),
        )
    })
}

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host
        .trim_matches(|character| character == '[' || character == ']')
        .parse::<IpAddr>()
    {
        Ok(address) => address.is_loopback(),
        Err(_) => false,
    }
}

fn validate_service_limits(limits: GatewayHttpLimits) -> GatewayResult<()> {
    if limits.max_request_bytes == 0 {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "max_request_bytes must be greater than zero",
        ));
    }
    Ok(())
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
}

#[derive(Serialize)]
struct StatusBody {
    uptime_secs: u64,
    aliases: Vec<AliasStatusBody>,
}

#[derive(Serialize)]
struct AliasStatusBody {
    name: String,
    operations: Vec<&'static str>,
    limits: AliasLimitsBody,
    global_total_requests: u64,
    global_window_requests: u32,
    tracked_callers: usize,
}

impl From<GatewayAliasSnapshot> for AliasStatusBody {
    fn from(snapshot: GatewayAliasSnapshot) -> Self {
        let mut operations = Vec::new();
        if snapshot.query {
            operations.push("query");
        }
        if snapshot.chat {
            operations.push("chat");
        }
        if snapshot.embed {
            operations.push("embed");
        }
        Self {
            name: snapshot.name,
            operations,
            limits: AliasLimitsBody::from(snapshot.limits),
            global_total_requests: snapshot.global_total_requests,
            global_window_requests: snapshot.global_window_requests,
            tracked_callers: snapshot.tracked_callers,
        }
    }
}

#[derive(Serialize)]
struct AliasLimitsBody {
    global: RequestLimitsBody,
    per_caller: Option<RequestLimitsBody>,
    max_tracked_callers: usize,
}

impl From<GatewayAliasLimits> for AliasLimitsBody {
    fn from(limits: GatewayAliasLimits) -> Self {
        Self {
            global: RequestLimitsBody::from(limits.global),
            per_caller: limits.per_caller.map(RequestLimitsBody::from),
            max_tracked_callers: limits.max_tracked_callers,
        }
    }
}

#[derive(Serialize)]
struct RequestLimitsBody {
    max_concurrent_requests: Option<usize>,
    max_requests_per_minute: Option<u32>,
    max_requests_total: Option<u64>,
}

impl From<cogentlm_gateway::GatewayRequestLimits> for RequestLimitsBody {
    fn from(limits: cogentlm_gateway::GatewayRequestLimits) -> Self {
        Self {
            max_concurrent_requests: limits.max_concurrent_requests,
            max_requests_per_minute: limits.max_requests_per_minute,
            max_requests_total: limits.max_requests_total,
        }
    }
}

#[derive(Serialize)]
struct HistoryBody {
    entries: Vec<HistoryEntry>,
}

const DASHBOARD_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CogentLM Gateway</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }
    input, textarea, select, button { font: inherit; margin: .25rem 0; }
    input, textarea { width: min(720px, 100%); box-sizing: border-box; }
    textarea { min-height: 7rem; }
    pre { background: #111827; color: #f9fafb; padding: 1rem; overflow: auto; }
    section { margin-block: 1.5rem; }
    label { display: block; font-weight: 600; margin-top: .75rem; }
  </style>
</head>
<body>
  <h1>CogentLM Gateway</h1>
  <p>This standalone server exposes <code>/v1/query</code>, <code>/v1/chat</code>, and <code>/v1/embed</code>. Status and history require the admin token.</p>

  <section>
    <h2>Status</h2>
    <label>Admin token <input id="admin-token" type="password" autocomplete="off"></label>
    <button id="refresh">Refresh status and history</button>
    <pre id="status">Enter the admin token and refresh.</pre>
  </section>

  <section>
    <h2>Manual Trigger</h2>
    <label>Inference token <input id="inference-token" type="password" autocomplete="off"></label>
    <label>Alias <input id="alias" value="local"></label>
    <label>Operation
      <select id="operation">
        <option value="query">query</option>
        <option value="chat">chat</option>
        <option value="embed">embed</option>
      </select>
    </label>
    <label>Input <textarea id="input">Write one sentence about gateway inference.</textarea></label>
    <button id="run">Run</button>
    <pre id="output">Manual responses appear here.</pre>
  </section>

  <section>
    <h2>curl</h2>
    <pre>curl -s http://127.0.0.1:8787/v1/query \
  -H "Authorization: Bearer $COGENTLM_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","prompt":"hello","max_tokens":32}'</pre>
  </section>

  <script>
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    const adminToken = document.getElementById('admin-token');
    const inferenceToken = document.getElementById('inference-token');
    const alias = document.getElementById('alias');
    const operation = document.getElementById('operation');
    const input = document.getElementById('input');

    async function adminJson(path) {
      const response = await fetch(path, {
        headers: { Authorization: `Bearer ${adminToken.value}` },
      });
      const text = await response.text();
      try { return JSON.parse(text); } catch { return { status: response.status, body: text }; }
    }

    document.getElementById('refresh').onclick = async () => {
      statusEl.textContent = 'Loading...';
      const [status, history] = await Promise.all([
        adminJson('/admin/api/status'),
        adminJson('/admin/api/history'),
      ]);
      statusEl.textContent = JSON.stringify({ status, history }, null, 2);
    };

    document.getElementById('run').onclick = async () => {
      outputEl.textContent = 'Running...';
      const op = operation.value;
      const body = op === 'chat'
        ? { model: alias.value, messages: [{ role: 'user', content: input.value }], max_tokens: 64 }
        : op === 'embed'
          ? { model: alias.value, input: input.value }
          : { model: alias.value, prompt: input.value, max_tokens: 64 };
      const response = await fetch(`/v1/${op}`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${inferenceToken.value}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });
      outputEl.textContent = await response.text();
    };
  </script>
</body>
</html>
"#;
