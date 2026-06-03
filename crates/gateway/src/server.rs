use std::{
    collections::BTreeMap,
    convert::Infallible,
    fmt,
    net::IpAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, Extension, Request, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER},
        HeaderMap, HeaderValue, Method,
    },
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::post,
    Json, Router,
};
use futures_util::{Stream, StreamExt};
use serde_json::json;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower_http::cors::CorsLayer;

use crate::{
    protocol::{
        finish_reason, validate_gateway_options, validate_non_empty, validate_text_options,
        EmbedRequestBody, EmbeddingResponseBody, QueryRequestBody, TextResponseBody, UsageBody,
    },
    BackendEmbeddingOutput, BackendTextOutput, GatewayBackend, GatewayError, GatewayErrorKind,
    GatewayResult, GatewayStream, GatewayStreamEvent, Operation, OperationSet,
};

const X_REQUEST_ID: axum::http::HeaderName = axum::http::HeaderName::from_static("x-request-id");
const RETRY_AFTER_MS: axum::http::HeaderName =
    axum::http::HeaderName::from_static("retry-after-ms");
static NEXT_GATEWAY_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Default maximum request body size accepted by the gateway.
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 1 << 20;

/// Gateway-wide HTTP service limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayServiceLimits {
    /// Maximum JSON request body size in bytes.
    pub max_request_bytes: usize,
}

impl Default for GatewayServiceLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
        }
    }
}

/// Alias-specific gateway policy limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GatewayAliasLimits {
    /// Maximum concurrent requests for this alias.
    pub max_concurrent_requests: Option<usize>,
    /// Maximum requests per rolling minute for this alias.
    pub max_requests_per_minute: Option<u32>,
    /// Maximum total requests allowed since gateway startup.
    pub max_requests_total: Option<u64>,
}

/// Public gateway alias and its server-side backend.
#[derive(Clone)]
pub struct GatewayAlias {
    name: String,
    operations: OperationSet,
    backend: Arc<dyn GatewayBackend>,
    limits: GatewayAliasLimits,
    concurrency: Option<Arc<Semaphore>>,
}

impl GatewayAlias {
    /// Create a public alias backed by a server-side backend.
    pub fn new(
        name: impl Into<String>,
        operations: OperationSet,
        backend: Arc<dyn GatewayBackend>,
        limits: GatewayAliasLimits,
    ) -> Self {
        let concurrency = limits
            .max_concurrent_requests
            .map(|limit| Arc::new(Semaphore::new(limit)));
        Self {
            name: name.into(),
            operations,
            backend,
            limits,
            concurrency,
        }
    }
}

/// Access scope attached to a gateway token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayAccess {
    aliases: Option<BTreeMap<String, OperationSet>>,
}

impl GatewayAccess {
    /// Allow every alias and operation.
    pub const fn all() -> Self {
        Self { aliases: None }
    }

    /// Restrict access to the listed aliases and operations.
    pub fn new(aliases: impl IntoIterator<Item = (String, OperationSet)>) -> Self {
        Self {
            aliases: Some(aliases.into_iter().collect()),
        }
    }

    fn allows(&self, alias: &str, operation: Operation) -> bool {
        match &self.aliases {
            None => true,
            Some(aliases) => aliases
                .get(alias)
                .is_some_and(|operations| operations.supports(operation)),
        }
    }
}

/// Gateway bearer token and its access scope.
#[derive(Clone, PartialEq, Eq)]
pub struct GatewayToken {
    secret: String,
    access: GatewayAccess,
}

impl GatewayToken {
    /// Create a scoped gateway token.
    pub fn new(secret: impl Into<String>, access: GatewayAccess) -> Self {
        Self {
            secret: secret.into(),
            access,
        }
    }
}

impl fmt::Debug for GatewayToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GatewayToken")
            .field("secret", &"[redacted]")
            .field("access", &self.access)
            .finish()
    }
}

/// Shared gateway HTTP state.
#[derive(Clone)]
pub struct GatewayState {
    tokens: Vec<GatewayToken>,
    aliases: BTreeMap<String, GatewayAlias>,
    policy: Arc<Mutex<GatewayPolicyState>>,
}

impl GatewayState {
    /// Create an empty gateway state with a required bearer token.
    pub fn new(token: impl Into<String>) -> Self {
        Self::with_tokens([GatewayToken::new(token, GatewayAccess::all())])
    }

    /// Create an empty gateway state with explicit bearer-token scopes.
    pub fn with_tokens(tokens: impl IntoIterator<Item = GatewayToken>) -> Self {
        Self {
            tokens: tokens.into_iter().collect(),
            aliases: BTreeMap::new(),
            policy: Arc::new(Mutex::new(GatewayPolicyState::default())),
        }
    }

    /// Add an alias.
    pub fn add_alias(&mut self, alias: GatewayAlias) -> GatewayResult<()> {
        validate_non_empty(&alias.name, "alias name")?;
        validate_alias_limits(alias.limits)?;
        if self.aliases.contains_key(&alias.name) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("duplicate gateway alias: {}", alias.name),
            ));
        }
        {
            let mut policy = self.policy_lock()?;
            if policy.aliases.contains_key(&alias.name) {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    format!("duplicate gateway alias policy: {}", alias.name),
                ));
            }
            policy.aliases.insert(
                alias.name.clone(),
                AliasPolicyState {
                    window_started: Instant::now(),
                    ..AliasPolicyState::default()
                },
            );
        }
        self.aliases.insert(alias.name.clone(), alias);
        Ok(())
    }

    fn authorize(&self, headers: &HeaderMap) -> GatewayResult<GatewayAccess> {
        if self.tokens.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::Authentication,
                "gateway has no configured bearer tokens",
            ));
        }
        let Some(header) = headers.get(AUTHORIZATION) else {
            return Err(GatewayError::new(
                GatewayErrorKind::Authentication,
                "missing bearer token",
            ));
        };
        let value = header.to_str().map_err(|_| {
            GatewayError::new(
                GatewayErrorKind::Authentication,
                "invalid authorization header",
            )
        })?;
        let Some(token) = value.strip_prefix("Bearer ") else {
            return Err(GatewayError::new(
                GatewayErrorKind::Authentication,
                "authorization header must use bearer auth",
            ));
        };
        for candidate in &self.tokens {
            validate_non_empty(&candidate.secret, "gateway token")?;
            if constant_time_eq(token.as_bytes(), candidate.secret.as_bytes()) {
                return Ok(candidate.access.clone());
            }
        }
        Err(GatewayError::new(
            GatewayErrorKind::Authentication,
            "invalid bearer token",
        ))
    }

    fn alias(
        &self,
        access: &GatewayAccess,
        model: &str,
        operation: Operation,
    ) -> GatewayResult<GatewayAlias> {
        validate_non_empty(model, "model")?;
        let alias = self.aliases.get(model).ok_or_else(|| {
            GatewayError::new(
                GatewayErrorKind::ModelNotFound,
                format!("model alias not found: {model}"),
            )
        })?;
        if !access.allows(model, operation) {
            return Err(GatewayError::new(
                GatewayErrorKind::Authorization,
                format!("token is not allowed to use {model} {}", operation.as_str()),
            ));
        }
        if !alias.operations.supports(operation) {
            return Err(GatewayError::new(
                GatewayErrorKind::UnsupportedFeature,
                format!("model alias does not support {}", operation.as_str()),
            ));
        }
        Ok(alias.clone())
    }

    fn acquire(&self, alias: &GatewayAlias) -> GatewayResult<GatewayRequestPermit> {
        let concurrency = match &alias.concurrency {
            Some(semaphore) => Some(semaphore.clone().try_acquire_owned().map_err(|_| {
                GatewayError::new(
                    GatewayErrorKind::Overloaded,
                    format!("model alias concurrency limit exceeded: {}", alias.name),
                )
            })?),
            None => None,
        };

        let mut policy = self.policy_lock()?;
        let state = policy.aliases.get_mut(&alias.name).ok_or_else(|| {
            GatewayError::new(
                GatewayErrorKind::Internal,
                format!("missing policy state for alias: {}", alias.name),
            )
        })?;
        state.check_and_increment(alias)?;
        Ok(GatewayRequestPermit {
            _concurrency: concurrency,
        })
    }

    fn policy_lock(&self) -> GatewayResult<std::sync::MutexGuard<'_, GatewayPolicyState>> {
        self.policy.lock().map_err(|_| {
            GatewayError::new(
                GatewayErrorKind::Internal,
                "gateway policy state lock is poisoned",
            )
        })
    }
}

#[derive(Default)]
struct GatewayPolicyState {
    aliases: BTreeMap<String, AliasPolicyState>,
}

struct AliasPolicyState {
    window_started: Instant,
    window_count: u32,
    total_count: u64,
}

impl Default for AliasPolicyState {
    fn default() -> Self {
        Self {
            window_started: Instant::now(),
            window_count: 0,
            total_count: 0,
        }
    }
}

impl AliasPolicyState {
    fn check_and_increment(&mut self, alias: &GatewayAlias) -> GatewayResult<()> {
        let now = Instant::now();
        if now.duration_since(self.window_started) >= Duration::from_secs(60) {
            self.window_started = now;
            self.window_count = 0;
        }

        if alias
            .limits
            .max_requests_total
            .is_some_and(|limit| self.total_count >= limit)
        {
            return Err(GatewayError::new(
                GatewayErrorKind::QuotaExceeded,
                format!("model alias quota exhausted: {}", alias.name),
            ));
        }

        if alias
            .limits
            .max_requests_per_minute
            .is_some_and(|limit| self.window_count >= limit)
        {
            let retry_after = Duration::from_secs(60)
                .saturating_sub(now.duration_since(self.window_started))
                .max(Duration::from_secs(1));
            return Err(GatewayError::new(
                GatewayErrorKind::RateLimited,
                format!("model alias rate limit exceeded: {}", alias.name),
            )
            .with_retry_after(Some(retry_after)));
        }

        self.total_count += 1;
        self.window_count += 1;
        Ok(())
    }
}

struct GatewayRequestPermit {
    _concurrency: Option<OwnedSemaphorePermit>,
}

#[derive(Clone)]
struct GatewayRequestId(String);

impl GatewayRequestId {
    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Gateway service with HTTP router settings.
#[derive(Clone)]
pub struct GatewayService {
    state: GatewayState,
    allowed_origins: Vec<String>,
    limits: GatewayServiceLimits,
}

impl GatewayService {
    /// Create a gateway service.
    pub fn new(
        state: GatewayState,
        allowed_origins: Vec<String>,
        limits: GatewayServiceLimits,
    ) -> Self {
        Self {
            state,
            allowed_origins,
            limits,
        }
    }

    /// Build the Axum router.
    pub fn router(self) -> GatewayResult<Router> {
        let state = Arc::new(self.state);
        let router = Router::new()
            .route("/v1/query", post(query))
            .route("/v1/chat", post(chat))
            .route("/v1/embed", post(embed))
            .with_state(state)
            .layer(DefaultBodyLimit::max(self.limits.max_request_bytes));

        if self.allowed_origins.is_empty() {
            return Ok(router.layer(middleware::from_fn(add_request_id)));
        }

        let mut origins = Vec::with_capacity(self.allowed_origins.len());
        for origin in self.allowed_origins {
            origins.push(cors_origin_header(&origin)?);
        }

        let router = router.layer(
            CorsLayer::new()
                .allow_methods([Method::POST, Method::OPTIONS])
                .allow_headers([AUTHORIZATION, CONTENT_TYPE])
                .expose_headers([X_REQUEST_ID.clone(), RETRY_AFTER, RETRY_AFTER_MS.clone()])
                .allow_origin(origins),
        );
        Ok(router.layer(middleware::from_fn(add_request_id)))
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

    let uri = trimmed.parse::<axum::http::Uri>().map_err(|error| {
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
        target: "cogentlm_gateway::request",
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

async fn query(
    State(state): State<Arc<GatewayState>>,
    Extension(request_id): Extension<GatewayRequestId>,
    headers: HeaderMap,
    body: Result<Json<QueryRequestBody>, JsonRejection>,
) -> Result<Response, GatewayError> {
    let started = Instant::now();
    let access = state.authorize(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    validate_gateway_options(&body.gateway_options)?;
    validate_non_empty(&body.prompt, "prompt")?;
    validate_text_options(&body.clone().into_backend().options)?;
    let alias = state.alias(&access, &body.model, Operation::Query)?;
    let permit = state.acquire(&alias)?;
    if body.stream {
        let model = body.model.clone();
        let stream = alias.backend.stream_query(body.into_backend()).await?;
        return Ok(sse(observe_stream(
            hold_permit(stream, permit),
            request_id,
            model,
            Operation::Query,
            started,
        ))
        .into_response());
    }
    let model = body.model.clone();
    let output = alias.backend.query(body.into_backend()).await?;
    log_text_success(&request_id, &model, Operation::Query, &output, started);
    drop(permit);
    Ok(Json(text_response(&request_id, model, output)).into_response())
}

async fn chat(
    State(state): State<Arc<GatewayState>>,
    Extension(request_id): Extension<GatewayRequestId>,
    headers: HeaderMap,
    body: Result<Json<crate::protocol::ChatRequestBody>, JsonRejection>,
) -> Result<Response, GatewayError> {
    let started = Instant::now();
    let access = state.authorize(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    validate_gateway_options(&body.gateway_options)?;
    if body.messages.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "messages must not be empty",
        ));
    }
    validate_text_options(&body.clone().into_backend().options)?;
    let alias = state.alias(&access, &body.model, Operation::Chat)?;
    let permit = state.acquire(&alias)?;
    if body.stream {
        let model = body.model.clone();
        let stream = alias.backend.stream_chat(body.into_backend()).await?;
        return Ok(sse(observe_stream(
            hold_permit(stream, permit),
            request_id,
            model,
            Operation::Chat,
            started,
        ))
        .into_response());
    }
    let model = body.model.clone();
    let output = alias.backend.chat(body.into_backend()).await?;
    log_text_success(&request_id, &model, Operation::Chat, &output, started);
    drop(permit);
    Ok(Json(text_response(&request_id, model, output)).into_response())
}

async fn embed(
    State(state): State<Arc<GatewayState>>,
    Extension(request_id): Extension<GatewayRequestId>,
    headers: HeaderMap,
    body: Result<Json<EmbedRequestBody>, JsonRejection>,
) -> Result<Response, GatewayError> {
    let started = Instant::now();
    let access = state.authorize(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    validate_gateway_options(&body.gateway_options)?;
    validate_non_empty(&body.input, "input")?;
    let alias = state.alias(&access, &body.model, Operation::Embed)?;
    let permit = state.acquire(&alias)?;
    let model = body.model.clone();
    let output = alias.backend.embed(body.into_backend()).await?;
    log_embedding_success(&request_id, &model, &output, started);
    drop(permit);
    Ok(Json(embedding_response(&request_id, model, output)).into_response())
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
        usage: output.usage.map(UsageBody::from),
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
        usage: output.usage.map(UsageBody::from),
    }
}

fn sse(
    stream: impl Stream<Item = GatewayResult<GatewayStreamEvent>> + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Sse::new(stream.map(|event| Ok(sse_event(event)))).keep_alive(KeepAlive::default())
}

fn hold_permit(
    stream: impl Stream<Item = GatewayResult<GatewayStreamEvent>> + Send + 'static,
    permit: GatewayRequestPermit,
) -> impl Stream<Item = GatewayResult<GatewayStreamEvent>> + Send + 'static {
    stream.map(move |event| {
        let _permit = &permit;
        event
    })
}

fn observe_stream(
    stream: impl Stream<Item = GatewayResult<GatewayStreamEvent>> + Send + 'static,
    request_id: GatewayRequestId,
    alias: String,
    operation: Operation,
    started: Instant,
) -> GatewayStream<GatewayStreamEvent> {
    let mut usage = None;
    let operation = operation.as_str();
    Box::pin(stream.map(move |event| {
        match &event {
            Ok(GatewayStreamEvent::Usage { usage: current }) => {
                usage = Some(*current);
            }
            Ok(GatewayStreamEvent::Finished { finish_reason }) => {
                log_gateway_operation(
                    request_id.as_str(),
                    &alias,
                    operation,
                    Some(finish_reason.as_str()),
                    usage,
                    started,
                );
            }
            Err(error) => {
                let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
                tracing::warn!(
                    target: "cogentlm_gateway::operation",
                    request_id = %request_id.as_str(),
                    alias = %alias,
                    operation,
                    error_code = %error.code,
                    latency_ms,
                    "gateway stream failed"
                );
            }
            Ok(GatewayStreamEvent::TokenBatch(_)) => {}
        }
        event
    }))
}

fn log_text_success(
    request_id: &GatewayRequestId,
    alias: &str,
    operation: Operation,
    output: &BackendTextOutput,
    started: Instant,
) {
    log_gateway_operation(
        request_id.as_str(),
        alias,
        operation.as_str(),
        Some(output.finish_reason.as_str()),
        output.usage,
        started,
    );
}

fn log_embedding_success(
    request_id: &GatewayRequestId,
    alias: &str,
    output: &BackendEmbeddingOutput,
    started: Instant,
) {
    log_gateway_operation(
        request_id.as_str(),
        alias,
        Operation::Embed.as_str(),
        None,
        output.usage,
        started,
    );
}

fn log_gateway_operation(
    request_id: &str,
    alias: &str,
    operation: &'static str,
    finish_reason: Option<&str>,
    usage: Option<cogentlm_core::TokenUsage>,
    started: Instant,
) {
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(
        target: "cogentlm_gateway::operation",
        request_id,
        alias,
        operation,
        finish_reason = ?finish_reason,
        input_tokens = ?usage.and_then(|usage| usage.input_tokens),
        output_tokens = ?usage.and_then(|usage| usage.output_tokens),
        total_tokens = ?usage.and_then(|usage| usage.total_tokens),
        latency_ms,
        "gateway operation"
    );
}

fn json_rejection_error(rejection: JsonRejection) -> GatewayError {
    if rejection.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        return GatewayError::new(
            GatewayErrorKind::RequestTooLarge,
            "request body exceeds gateway limit",
        );
    }
    GatewayError::new(GatewayErrorKind::InvalidRequest, rejection.body_text())
}

fn sse_event(event: GatewayResult<GatewayStreamEvent>) -> Event {
    match event {
        Ok(GatewayStreamEvent::TokenBatch(batch)) => Event::default()
            .event("token")
            .json_data(json!({
                "text": batch.text,
                "sequence": batch.sequence_start,
            }))
            .unwrap_or_else(internal_sse_error),
        Ok(GatewayStreamEvent::Usage { usage }) => Event::default()
            .event("usage")
            .json_data(UsageBody::from(usage))
            .unwrap_or_else(internal_sse_error),
        Ok(GatewayStreamEvent::Finished { finish_reason }) => Event::default()
            .event("done")
            .json_data(json!({ "finish_reason": finish_reason.as_str() }))
            .unwrap_or_else(internal_sse_error),
        Err(error) => Event::default()
            .event("error")
            .json_data(json!({
                "error": {
                    "code": error.code,
                    "message": error.message,
                }
            }))
            .unwrap_or_else(internal_sse_error),
    }
}

fn internal_sse_error(_: axum::Error) -> Event {
    Event::default()
        .event("error")
        .data(r#"{"error":{"code":"internal","message":"failed to encode SSE event"}}"#)
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let left = left.get(index).copied().unwrap_or_default();
        let right = right.get(index).copied().unwrap_or_default();
        diff |= usize::from(left ^ right);
    }
    diff == 0
}

fn validate_alias_limits(limits: GatewayAliasLimits) -> GatewayResult<()> {
    if matches!(limits.max_concurrent_requests, Some(0)) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "max_concurrent_requests must be greater than zero",
        ));
    }
    if matches!(limits.max_requests_per_minute, Some(0)) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "max_requests_per_minute must be greater than zero",
        ));
    }
    if matches!(limits.max_requests_total, Some(0)) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "max_requests_total must be greater than zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/server_tests.rs"]
mod server_tests;
