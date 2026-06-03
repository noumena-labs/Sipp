use std::{
    collections::{BTreeMap, BTreeSet},
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
use cogentlm_core::TokenUsage;
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
    ) -> GatewayResult<Self> {
        let name = name.into();
        validate_gateway_alias_name(&name)?;
        if operations.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway alias operations must not be empty",
            ));
        }
        validate_alias_limits(limits)?;
        let concurrency = limits
            .max_concurrent_requests
            .map(|limit| Arc::new(Semaphore::new(limit)));
        Ok(Self {
            name,
            operations,
            backend,
            limits,
            concurrency,
        })
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
    ///
    /// Returns an error when aliases are blank, duplicated, or mapped to no
    /// operations.
    pub fn new(aliases: impl IntoIterator<Item = (String, OperationSet)>) -> GatewayResult<Self> {
        let mut access = BTreeMap::new();
        for (alias, operations) in aliases {
            validate_gateway_access_alias(&alias)?;
            if operations.is_empty() {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    "token access operations must not be empty",
                ));
            }
            if access.insert(alias, operations).is_some() {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    "token access aliases must not contain duplicates",
                ));
            }
        }
        if access.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "token access aliases must not be empty",
            ));
        }
        Ok(Self {
            aliases: Some(access),
        })
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

fn validate_gateway_alias_name(alias: &str) -> GatewayResult<()> {
    if alias.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "alias name must not be empty",
        ));
    }
    if alias.trim() != alias {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "alias name must not contain surrounding whitespace",
        ));
    }
    Ok(())
}

fn validate_gateway_access_alias(alias: &str) -> GatewayResult<()> {
    if alias.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "token access alias name must not be empty",
        ));
    }
    if alias.trim() != alias {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "token access alias name must not contain surrounding whitespace",
        ));
    }
    Ok(())
}

/// Gateway bearer token and its access scope.
#[derive(Clone, PartialEq, Eq)]
pub struct GatewayToken {
    secret: String,
    access: GatewayAccess,
}

impl GatewayToken {
    /// Create a scoped gateway token.
    ///
    /// Returns an error when the token is blank or contains whitespace.
    pub fn new(secret: impl Into<String>, access: GatewayAccess) -> GatewayResult<Self> {
        let secret = secret.into();
        validate_gateway_bearer_secret(&secret, "gateway token")?;
        Ok(Self { secret, access })
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
    ///
    /// Returns an error when the token is blank or contains whitespace.
    pub fn new(token: impl Into<String>) -> GatewayResult<Self> {
        Self::with_tokens([GatewayToken::new(token, GatewayAccess::all())?])
    }

    /// Create an empty gateway state with explicit bearer-token scopes.
    ///
    /// Returns an error when no tokens are provided.
    pub fn with_tokens(tokens: impl IntoIterator<Item = GatewayToken>) -> GatewayResult<Self> {
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        if tokens.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway requires at least one bearer token",
            ));
        }
        let mut unique_tokens = BTreeSet::new();
        for token in &tokens {
            if !unique_tokens.insert(token.secret.as_str()) {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    "gateway bearer tokens must be unique",
                ));
            }
        }
        Ok(Self {
            tokens,
            aliases: BTreeMap::new(),
            policy: Arc::new(Mutex::new(GatewayPolicyState::default())),
        })
    }

    /// Add an alias.
    pub fn add_alias(&mut self, alias: GatewayAlias) -> GatewayResult<()> {
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

    fn authorize(&self, headers: &HeaderMap) -> GatewayResult<&GatewayAccess> {
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
        let mut matched_index = None;
        for (index, candidate) in self.tokens.iter().enumerate() {
            if constant_time_eq(token.as_bytes(), candidate.secret.as_bytes()) {
                matched_index = Some(index);
            }
        }
        matched_index
            .map(|index| &self.tokens[index].access)
            .ok_or_else(|| {
                GatewayError::new(GatewayErrorKind::Authentication, "invalid bearer token")
            })
    }

    fn alias(
        &self,
        access: &GatewayAccess,
        model: &str,
        operation: Operation,
    ) -> GatewayResult<&GatewayAlias> {
        self.authorize_alias_scope(access, model, operation)?;
        let alias = self.aliases.get(model).ok_or_else(|| {
            GatewayError::new(GatewayErrorKind::ModelNotFound, "model alias not found")
        })?;
        if !alias.operations.supports(operation) {
            return Err(GatewayError::new(
                GatewayErrorKind::UnsupportedFeature,
                format!("model alias does not support {}", operation.as_str()),
            ));
        }
        Ok(alias)
    }

    fn authorize_alias_scope(
        &self,
        access: &GatewayAccess,
        model: &str,
        operation: Operation,
    ) -> GatewayResult<()> {
        validate_non_empty(model, "model")?;
        if access.allows(model, operation) {
            return Ok(());
        }
        Err(GatewayError::new(
            GatewayErrorKind::Authorization,
            "token is not allowed to use the requested model operation",
        ))
    }

    fn acquire(&self, alias: &GatewayAlias) -> GatewayResult<GatewayRequestPermit> {
        let concurrency = match &alias.concurrency {
            Some(semaphore) => Some(semaphore.clone().try_acquire_owned().map_err(|_| {
                GatewayError::new(
                    GatewayErrorKind::Overloaded,
                    "model alias concurrency limit exceeded",
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
                "model alias quota exhausted",
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
                "model alias rate limit exceeded",
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
    allowed_origins: Vec<HeaderValue>,
    limits: GatewayServiceLimits,
}

impl GatewayService {
    /// Create a gateway service.
    ///
    /// Returns an error when service limits or CORS origins are invalid.
    pub fn new(
        state: GatewayState,
        allowed_origins: Vec<String>,
        limits: GatewayServiceLimits,
    ) -> GatewayResult<Self> {
        validate_service_limits(limits)?;
        let mut origins = Vec::with_capacity(allowed_origins.len());
        for origin in allowed_origins {
            origins.push(cors_origin_header(&origin)?);
        }
        Ok(Self {
            state,
            allowed_origins: origins,
            limits,
        })
    }

    /// Build the Axum router.
    pub fn router(self) -> Router {
        let state = Arc::new(self.state);
        let router = Router::new()
            .route("/v1/query", post(query))
            .route("/v1/chat", post(chat))
            .route("/v1/embed", post(embed))
            .with_state(state)
            .layer(DefaultBodyLimit::max(self.limits.max_request_bytes));

        if self.allowed_origins.is_empty() {
            return router.layer(middleware::from_fn(add_request_id));
        }

        let router = router.layer(
            CorsLayer::new()
                .allow_methods([Method::POST, Method::OPTIONS])
                .allow_headers([AUTHORIZATION, CONTENT_TYPE])
                .expose_headers([X_REQUEST_ID.clone(), RETRY_AFTER, RETRY_AFTER_MS.clone()])
                .allow_origin(self.allowed_origins),
        );
        router.layer(middleware::from_fn(add_request_id))
    }
}

pub(crate) fn cors_origin_header(origin: &str) -> GatewayResult<HeaderValue> {
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

pub(crate) fn is_loopback_host(host: &str) -> bool {
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
    state.authorize_alias_scope(access, &body.model, Operation::Query)?;
    validate_gateway_options(&body.gateway_options)?;
    validate_non_empty(&body.prompt, "prompt")?;
    validate_text_options(&body.generation_options())?;
    let alias = state.alias(access, &body.model, Operation::Query)?;
    let permit = state.acquire(alias)?;
    let backend = alias.backend.clone();
    if body.stream {
        let model = body.model.clone();
        let stream = backend.stream_query(body.into_backend()).await?;
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
    let output = backend.query(body.into_backend()).await?;
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
    state.authorize_alias_scope(access, &body.model, Operation::Chat)?;
    validate_gateway_options(&body.gateway_options)?;
    if body.messages.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "messages must not be empty",
        ));
    }
    validate_text_options(&body.generation_options())?;
    let alias = state.alias(access, &body.model, Operation::Chat)?;
    let permit = state.acquire(alias)?;
    let backend = alias.backend.clone();
    if body.stream {
        let model = body.model.clone();
        let stream = backend.stream_chat(body.into_backend()).await?;
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
    let output = backend.chat(body.into_backend()).await?;
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
    state.authorize_alias_scope(access, &body.model, Operation::Embed)?;
    validate_gateway_options(&body.gateway_options)?;
    validate_non_empty(&body.input, "input")?;
    let alias = state.alias(access, &body.model, Operation::Embed)?;
    let permit = state.acquire(alias)?;
    let backend = alias.backend.clone();
    let model = body.model.clone();
    let output = backend.embed(body.into_backend()).await?;
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
                    error_code = %error.code(),
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
    GatewayError::new(
        GatewayErrorKind::InvalidRequest,
        "invalid JSON request body",
    )
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

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let left = left.get(index).copied().unwrap_or_default();
        let right = right.get(index).copied().unwrap_or_default();
        diff |= usize::from(left ^ right);
    }
    diff == 0
}

pub(crate) fn validate_gateway_bearer_secret(secret: &str, field: &str) -> GatewayResult<()> {
    if secret.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not be empty"),
        ));
    }
    if secret.chars().any(char::is_whitespace) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not contain whitespace"),
        ));
    }
    Ok(())
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

fn validate_service_limits(limits: GatewayServiceLimits) -> GatewayResult<()> {
    if limits.max_request_bytes == 0 {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "max_request_bytes must be greater than zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/server_tests.rs"]
mod server_tests;
