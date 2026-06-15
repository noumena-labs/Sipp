use std::collections::VecDeque;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, Method, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use futures_util::future::{select, Either};
use futures_util::{stream, Stream, StreamExt};
use sipp::core::TokenUsage;
use sipp::gateway_core::{GatewayStreamEvent, Operation};
use sipp::{SippRequestContext, SippTextResponseFuture, SippTokenBatches};
use sipp_gateway::{
    request_context, request_id, AuthenticatedRequest, Authenticator, GatewayCodec,
    GatewayHttpError, GatewayRoutes, ProtocolCodec, ToolkitResult,
};
use tower_http::cors::CorsLayer;

use crate::admin::{self, AdminDashboardState, AdminDashboardView};
use crate::config::{GatewayServerRuntime, LoadedToken, RouteConfig, SecurityConfig};
use crate::metrics::{GatewayMetrics, GatewayRejectionKind, GatewayRequestRecord};
use crate::runtime::{
    ConcurrencyAcquireError, ConcurrencyPermit, GatewayControls, GatewaySecurity,
    SecurityCheckError,
};

/// Standalone application HTTP composition.
pub struct GatewayHttpService {
    public: Router,
    management: Router,
}

impl GatewayHttpService {
    /// Compose public and management routers from application-owned handlers.
    pub fn new(
        runtime: GatewayServerRuntime,
        routes: RouteConfig,
        tokens: Vec<LoadedToken>,
        admin_password: String,
        metrics: Arc<GatewayMetrics>,
        max_request_bytes: usize,
        allowed_origins: &[String],
        max_concurrent_requests: Option<usize>,
        security_config: SecurityConfig,
        admin_assets_dir: PathBuf,
    ) -> anyhow::Result<Self> {
        let gateway_routes: GatewayRoutes = routes.clone().into();
        let controls = Arc::new(GatewayControls::new(max_concurrent_requests));
        let security = Arc::new(GatewaySecurity::new(&security_config)?);
        let state = PublicState {
            runtime: runtime.clone(),
            authenticator: Arc::new(BearerAuthenticator { tokens }),
            metrics: metrics.clone(),
            controls: controls.clone(),
            security: security.clone(),
            codec: GatewayCodec,
        };
        let mut public = Router::new()
            .route(&gateway_routes.query, post(query))
            .route(&gateway_routes.chat, post(chat))
            .route(&gateway_routes.embed, post(embed))
            .with_state(state)
            .layer(axum::extract::DefaultBodyLimit::max(max_request_bytes));
        if !allowed_origins.is_empty() {
            let origins = allowed_origins
                .iter()
                .map(|origin| HeaderValue::from_str(origin))
                .collect::<Result<Vec<_>, _>>()?;
            public = public.layer(
                CorsLayer::new()
                    .allow_methods([Method::POST, Method::OPTIONS])
                    .allow_headers([AUTHORIZATION, CONTENT_TYPE])
                    .allow_origin(origins),
            );
        }

        let mut management = Router::new();
        if let Some(route) = gateway_routes.health {
            management = management.route(&route, get(health));
        }
        if let Some(route) = gateway_routes.readiness {
            management = management.route(&route, get(readiness));
        }
        if let Some(route) = gateway_routes.metrics {
            let metrics = metrics.clone();
            management = management.route(
                &route,
                get(move || {
                    let metrics = metrics.clone();
                    async move {
                        match metrics.render() {
                            Ok(body) => response(
                                StatusCode::OK,
                                "text/plain; version=0.0.4",
                                Body::from(body),
                                None,
                            ),
                            Err(error) => response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "text/plain; charset=utf-8",
                                Body::from(error),
                                None,
                            ),
                        }
                    }
                }),
            );
        }
        if let Some(route) = gateway_routes.index {
            management = management.route(&route, get(index));
        }
        if let Some(route) = routes.admin.as_deref() {
            let view = AdminDashboardView {
                routes: routes.clone(),
                targets: runtime.target_summaries.as_ref().clone(),
                max_request_bytes,
                max_concurrent_requests,
                started_at: std::time::Instant::now(),
            };
            management = management.merge(admin::router(
                route,
                AdminDashboardState::new(
                    admin_password,
                    view,
                    metrics,
                    controls,
                    security,
                    admin_assets_dir,
                )?,
            ));
        }
        Ok(Self { public, management })
    }

    /// Public inference router.
    pub fn public_router(&self) -> Router {
        self.public.clone()
    }

    /// Application management router.
    pub fn management_router(&self) -> Router {
        self.management.clone()
    }
}

#[derive(Clone)]
struct PublicState {
    runtime: GatewayServerRuntime,
    authenticator: Arc<dyn Authenticator>,
    metrics: Arc<GatewayMetrics>,
    controls: Arc<GatewayControls>,
    security: Arc<GatewaySecurity>,
    codec: GatewayCodec,
}

async fn query(
    State(state): State<PublicState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    text_handler(state, Operation::Query, peer, headers, body).await
}

async fn chat(
    State(state): State<PublicState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    text_handler(state, Operation::Chat, peer, headers, body).await
}

async fn embed(
    State(state): State<PublicState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let request_id = request_id(&headers);
    let trace = RequestTrace::new(&state, Operation::Embed, request_id, peer, &headers, false);
    state
        .metrics
        .request_started(Operation::Embed, trace.request_id());
    if let Err(rejection) = state.security.check_client(&trace.client) {
        return security_rejection_response(&state, trace, rejection);
    }
    let authenticated = match state.authenticator.authenticate(&headers) {
        Ok(authenticated) => authenticated,
        Err(error) => return error_response(&state, trace, None, None, error),
    };
    let caller = caller_label(&authenticated);
    let context = request_context(request_id, authenticated.clone());
    let decoded = match state.codec.decode_embed(&body) {
        Ok(decoded) => decoded,
        Err(error) => return error_response(&state, trace, caller, None, error),
    };
    let target = decoded.target.clone();
    if let Err(error) = authorize(&context, &decoded.target) {
        return error_response(&state, trace, caller, Some(target), error);
    }
    let endpoint = match resolve_endpoint(&state, &decoded.target) {
        Ok(endpoint) => endpoint,
        Err(error) => return error_response(&state, trace, caller, Some(target), error),
    };
    let _permit = match acquire(&state) {
        Ok(permit) => permit,
        Err(error) => return error_response(&state, trace, caller, Some(target), error),
    };
    let mut request = decoded.request;
    request.endpoint = Some(endpoint);
    let run = state.runtime.client.embed_with_context(
        SippRequestContext {
            request_id: request_id.map(str::to_string),
        },
        request,
    );
    match run.await {
        Ok(response) => match state.codec.encode_embedding(&target, &response) {
            Ok(body) => success_response(
                &state,
                trace,
                caller,
                Some(target),
                false,
                response.usage,
                body,
            ),
            Err(error) => error_response(&state, trace, caller, Some(target), error),
        },
        Err(error) => error_response(
            &state,
            trace,
            caller,
            Some(target),
            GatewayHttpError::from_gateway_error(error.into()),
        ),
    }
}

async fn text_handler(
    state: PublicState,
    operation: Operation,
    peer: SocketAddr,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let request_id = request_id(&headers);
    let trace = RequestTrace::new(&state, operation, request_id, peer, &headers, false);
    state.metrics.request_started(operation, trace.request_id());
    if let Err(rejection) = state.security.check_client(&trace.client) {
        return security_rejection_response(&state, trace, rejection);
    }
    let authenticated = match state.authenticator.authenticate(&headers) {
        Ok(authenticated) => authenticated,
        Err(error) => return error_response(&state, trace, None, None, error),
    };
    let caller = caller_label(&authenticated);
    let context = request_context(request_id, authenticated.clone());
    match operation {
        Operation::Query => {
            let decoded = match state.codec.decode_query(&body) {
                Ok(decoded) => decoded,
                Err(error) => return error_response(&state, trace, caller, None, error),
            };
            let target = decoded.target.clone();
            if let Err(error) = authorize(&context, &decoded.target) {
                return error_response(&state, trace, caller, Some(target), error);
            }
            let endpoint = match resolve_endpoint(&state, &decoded.target) {
                Ok(endpoint) => endpoint,
                Err(error) => return error_response(&state, trace, caller, Some(target), error),
            };
            let permit = match acquire(&state) {
                Ok(permit) => permit,
                Err(error) => return error_response(&state, trace, caller, Some(target), error),
            };
            let mut request = decoded.request;
            request.endpoint = Some(endpoint);
            request.emit_tokens = decoded.stream;
            let run = state.runtime.client.query_with_context(
                SippRequestContext {
                    request_id: request_id.map(str::to_string),
                },
                request,
            );
            if decoded.stream {
                stream_response(
                    &state,
                    trace.into_streaming(),
                    caller,
                    Some(target),
                    run.into_parts(),
                    permit,
                )
            } else {
                let _permit = permit;
                match run.await {
                    Ok(response) => match state.codec.encode_text(&target, &response) {
                        Ok(body) => success_response(
                            &state,
                            trace,
                            caller,
                            Some(target),
                            false,
                            response.usage,
                            body,
                        ),
                        Err(error) => error_response(&state, trace, caller, Some(target), error),
                    },
                    Err(error) => error_response(
                        &state,
                        trace,
                        caller,
                        Some(target),
                        GatewayHttpError::from_gateway_error(error.into()),
                    ),
                }
            }
        }
        Operation::Chat => {
            let decoded = match state.codec.decode_chat(&body) {
                Ok(decoded) => decoded,
                Err(error) => return error_response(&state, trace, caller, None, error),
            };
            let target = decoded.target.clone();
            if let Err(error) = authorize(&context, &decoded.target) {
                return error_response(&state, trace, caller, Some(target), error);
            }
            let endpoint = match resolve_endpoint(&state, &decoded.target) {
                Ok(endpoint) => endpoint,
                Err(error) => return error_response(&state, trace, caller, Some(target), error),
            };
            let permit = match acquire(&state) {
                Ok(permit) => permit,
                Err(error) => return error_response(&state, trace, caller, Some(target), error),
            };
            let mut request = decoded.request;
            request.endpoint = Some(endpoint);
            request.emit_tokens = decoded.stream;
            let run = state.runtime.client.chat_with_context(
                SippRequestContext {
                    request_id: request_id.map(str::to_string),
                },
                request,
            );
            if decoded.stream {
                stream_response(
                    &state,
                    trace.into_streaming(),
                    caller,
                    Some(target),
                    run.into_parts(),
                    permit,
                )
            } else {
                let _permit = permit;
                match run.await {
                    Ok(response) => match state.codec.encode_text(&target, &response) {
                        Ok(body) => success_response(
                            &state,
                            trace,
                            caller,
                            Some(target),
                            false,
                            response.usage,
                            body,
                        ),
                        Err(error) => error_response(&state, trace, caller, Some(target), error),
                    },
                    Err(error) => error_response(
                        &state,
                        trace,
                        caller,
                        Some(target),
                        GatewayHttpError::from_gateway_error(error.into()),
                    ),
                }
            }
        }
        Operation::Embed => unreachable!("embed uses its dedicated handler"),
    }
}

struct RequestTrace {
    operation: Operation,
    request_id: Option<String>,
    client: String,
    started_at: Instant,
    streaming: bool,
}

impl RequestTrace {
    fn new(
        state: &PublicState,
        operation: Operation,
        request_id: Option<&str>,
        peer: SocketAddr,
        headers: &HeaderMap,
        streaming: bool,
    ) -> Self {
        Self {
            operation,
            request_id: request_id.map(str::to_string),
            client: state.security.client_identity(headers, peer),
            started_at: Instant::now(),
            streaming,
        }
    }

    fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    fn into_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    fn record(
        &self,
        state: &PublicState,
        caller: Option<String>,
        target: Option<String>,
        status: StatusCode,
        usage: Option<TokenUsage>,
        rejection: Option<GatewayRejectionKind>,
    ) {
        self.record_with_metrics(&state.metrics, caller, target, status, usage, rejection);
    }

    fn record_with_metrics(
        &self,
        metrics: &GatewayMetrics,
        caller: Option<String>,
        target: Option<String>,
        status: StatusCode,
        usage: Option<TokenUsage>,
        rejection: Option<GatewayRejectionKind>,
    ) {
        metrics.request_finished(GatewayRequestRecord {
            operation: self.operation,
            request_id: self.request_id.clone(),
            client: self.client.clone(),
            caller,
            target,
            status,
            duration: self.started_at.elapsed(),
            usage,
            streaming: self.streaming,
            rejection,
        });
    }
}

fn caller_label(authenticated: &AuthenticatedRequest) -> Option<String> {
    authenticated
        .metadata
        .get("caller")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn resolve_endpoint(state: &PublicState, target: &str) -> ToolkitResult<sipp::EndpointRef> {
    state.runtime.targets.get(target).cloned().ok_or_else(|| {
        GatewayHttpError::new(StatusCode::NOT_FOUND, "resolution", "target not found")
    })
}

fn authorize(
    context: &sipp::gateway_core::GatewayRequestContext,
    target: &str,
) -> ToolkitResult<()> {
    let allowed = context
        .metadata
        .get("targets")
        .and_then(serde_json::Value::as_array);
    if allowed.is_none_or(|allowed| {
        allowed.is_empty() || allowed.iter().any(|value| value.as_str() == Some(target))
    }) {
        Ok(())
    } else {
        Err(GatewayHttpError::new(
            StatusCode::FORBIDDEN,
            "authorization",
            "caller is not allowed to use the target",
        ))
    }
}

fn acquire(state: &PublicState) -> ToolkitResult<ConcurrencyPermit> {
    state.controls.try_acquire().map_err(|error| match error {
        ConcurrencyAcquireError::LimitExceeded => GatewayHttpError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "admission",
            "application concurrency limit exceeded",
        ),
        ConcurrencyAcquireError::Unavailable => GatewayHttpError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "admission",
            "application concurrency controls are unavailable",
        ),
    })
}

fn stream_response(
    state: &PublicState,
    trace: RequestTrace,
    caller: Option<String>,
    target: Option<String>,
    run: (SippTokenBatches, SippTextResponseFuture),
    permit: ConcurrencyPermit,
) -> Response<Body> {
    let response_request_id = trace.request_id.clone();
    let stream = text_event_stream(
        run,
        permit,
        StreamTelemetry {
            metrics: state.metrics.clone(),
            trace,
            caller,
            target,
        },
    )
    .map({
        let codec = state.codec;
        move |event| {
            let bytes = match event {
                Ok(event) => codec
                    .encode_stream_event(&event)
                    .unwrap_or_else(|error| codec.encode_stream_error(&error)),
                Err(error) => codec.encode_stream_error(&error),
            };
            Ok::<Bytes, Infallible>(bytes)
        }
    });
    response(
        StatusCode::OK,
        state.codec.content_type(true),
        Body::from_stream(stream),
        response_request_id.as_deref(),
    )
}

struct TextStreamState {
    tokens: SippTokenBatches,
    response: Option<SippTextResponseFuture>,
    pending: VecDeque<ToolkitResult<GatewayStreamEvent>>,
    terminal: bool,
    recorded: bool,
    telemetry: StreamTelemetry,
    _permit: Option<ConcurrencyPermit>,
}

struct StreamTelemetry {
    metrics: Arc<GatewayMetrics>,
    trace: RequestTrace,
    caller: Option<String>,
    target: Option<String>,
}

fn text_event_stream(
    (tokens, response): (SippTokenBatches, SippTextResponseFuture),
    permit: ConcurrencyPermit,
    telemetry: StreamTelemetry,
) -> impl Stream<Item = ToolkitResult<GatewayStreamEvent>> + Send {
    let state = TextStreamState {
        tokens,
        response: Some(response),
        pending: VecDeque::new(),
        terminal: false,
        recorded: false,
        telemetry,
        _permit: Some(permit),
    };
    stream::unfold(state, |mut state| async move {
        if let Some(event) = state.pending.pop_front() {
            return Some((event, state));
        }
        if state.terminal {
            return None;
        }
        let response = state.response.take()?;
        match select(state.tokens.next(), response).await {
            Either::Left((Some(batch), response)) => {
                state.response = Some(response);
                Some((Ok(GatewayStreamEvent::TokenBatch(batch)), state))
            }
            Either::Left((None, response)) => {
                finish_text_stream(&mut state, response.await);
                state.pending.pop_front().map(|event| (event, state))
            }
            Either::Right((response, tokens)) => {
                drop(tokens);
                finish_text_stream(&mut state, response);
                state.pending.pop_front().map(|event| (event, state))
            }
        }
    })
}

fn finish_text_stream(
    state: &mut TextStreamState,
    response: sipp::SippResult<sipp::SippTextResponse>,
) {
    state.terminal = true;
    state._permit.take();
    match response {
        Ok(response) => {
            state.record(StatusCode::OK, response.usage, None);
            if let Some(usage) = response.usage {
                state
                    .pending
                    .push_back(Ok(GatewayStreamEvent::Usage(usage)));
            }
            state.pending.push_back(Ok(GatewayStreamEvent::Finished {
                finish_reason: response.finish_reason,
                metadata: response.metadata,
            }));
        }
        Err(error) => {
            let error = GatewayHttpError::from_gateway_error(error.into());
            state.record(error.status, None, None);
            state.pending.push_back(Err(error));
        }
    }
}

impl TextStreamState {
    fn record(
        &mut self,
        status: StatusCode,
        usage: Option<TokenUsage>,
        rejection: Option<GatewayRejectionKind>,
    ) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        self.telemetry.trace.record_with_metrics(
            &self.telemetry.metrics,
            self.telemetry.caller.clone(),
            self.telemetry.target.clone(),
            status,
            usage,
            rejection,
        );
    }
}

impl Drop for TextStreamState {
    fn drop(&mut self) {
        if !self.recorded {
            self.record(
                StatusCode::REQUEST_TIMEOUT,
                None,
                Some(GatewayRejectionKind::ClientClosed),
            );
        }
    }
}

fn success_response(
    state: &PublicState,
    trace: RequestTrace,
    caller: Option<String>,
    target: Option<String>,
    streaming: bool,
    usage: Option<TokenUsage>,
    body: Bytes,
) -> Response<Body> {
    trace.record(state, caller, target, StatusCode::OK, usage, None);
    response(
        StatusCode::OK,
        state.codec.content_type(streaming),
        Body::from(body),
        trace.request_id(),
    )
}

fn error_response(
    state: &PublicState,
    trace: RequestTrace,
    caller: Option<String>,
    target: Option<String>,
    error: GatewayHttpError,
) -> Response<Body> {
    trace.record(state, caller, target, error.status, None, None);
    let body = state.codec.encode_error(&error);
    response(
        error.status,
        state.codec.content_type(false),
        Body::from(body),
        trace.request_id(),
    )
}

fn security_rejection_response(
    state: &PublicState,
    trace: RequestTrace,
    rejection: SecurityCheckError,
) -> Response<Body> {
    let (status, code, message, kind) = match rejection {
        SecurityCheckError::Blocked => (
            StatusCode::FORBIDDEN,
            "blocked",
            "client is blocked by the gateway",
            Some(GatewayRejectionKind::Blocked),
        ),
        SecurityCheckError::RateLimited => (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limited",
            "client rate limit exceeded",
            Some(GatewayRejectionKind::RateLimited),
        ),
        SecurityCheckError::Unavailable => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "security_controls",
            "gateway security controls are unavailable",
            None,
        ),
    };
    let error = GatewayHttpError::new(status, code, message);
    trace.record(state, None, None, status, None, kind);
    let body = state.codec.encode_error(&error);
    response(
        status,
        state.codec.content_type(false),
        Body::from(body),
        trace.request_id(),
    )
}

fn response(
    status: StatusCode,
    content_type: &'static str,
    body: Body,
    request_id: Option<&str>,
) -> Response<Body> {
    let mut builder = Response::builder()
        .status(status)
        .header("content-type", content_type);
    if let Some(request_id) = request_id.and_then(|value| HeaderValue::from_str(value).ok()) {
        builder = builder.header("x-request-id", request_id);
    }
    match builder.body(body) {
        Ok(response) => response,
        Err(_) => {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            response
        }
    }
}

struct BearerAuthenticator {
    tokens: Vec<LoadedToken>,
}

impl Authenticator for BearerAuthenticator {
    fn authenticate(&self, headers: &HeaderMap) -> ToolkitResult<AuthenticatedRequest> {
        let token = headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .ok_or_else(|| {
                GatewayHttpError::new(
                    StatusCode::UNAUTHORIZED,
                    "authentication",
                    "missing bearer token",
                )
            })?;
        let configured = self
            .tokens
            .iter()
            .find(|configured| constant_time_eq(token, &configured.secret))
            .ok_or_else(|| {
                GatewayHttpError::new(
                    StatusCode::UNAUTHORIZED,
                    "authentication",
                    "invalid bearer token",
                )
            })?;
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            "caller".to_string(),
            serde_json::Value::String(configured.caller.clone()),
        );
        metadata.insert(
            "targets".to_string(),
            serde_json::Value::Array(
                configured
                    .targets
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
        Ok(AuthenticatedRequest { metadata })
    }
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn readiness() -> impl IntoResponse {
    (StatusCode::OK, "ready")
}

async fn index() -> impl IntoResponse {
    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "capabilities": ["query", "chat", "embed"]
        })),
    )
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.bytes()
        .zip(right.bytes())
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}
