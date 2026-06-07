use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, Method, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use cogentlm_client::{CogentRequestContext, CogentTextResponseFuture, CogentTokenBatches};
use cogentlm_gateway::{
    request_context, request_id, AuthenticatedRequest, Authenticator, GatewayCodec,
    GatewayHttpError, GatewayObservability, GatewayRoutes, ProtocolCodec, ToolkitResult,
};
use cogentlm_gateway_core::{GatewayStreamEvent, Operation};
use futures_util::future::{select, Either};
use futures_util::{stream, Stream, StreamExt};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower_http::cors::CorsLayer;

use crate::config::{GatewayServerRuntime, LoadedToken};
use crate::metrics::GatewayMetrics;

/// Standalone application HTTP composition.
pub struct GatewayHttpService {
    public: Router,
    management: Router,
}

impl GatewayHttpService {
    /// Compose public and management routers from application-owned handlers.
    pub fn new(
        runtime: GatewayServerRuntime,
        routes: GatewayRoutes,
        tokens: Vec<LoadedToken>,
        metrics: Arc<GatewayMetrics>,
        max_request_bytes: usize,
        allowed_origins: &[String],
        max_concurrent_requests: Option<usize>,
    ) -> anyhow::Result<Self> {
        let state = PublicState {
            runtime,
            authenticator: Arc::new(BearerAuthenticator { tokens }),
            metrics: metrics.clone(),
            semaphore: max_concurrent_requests.map(|limit| Arc::new(Semaphore::new(limit))),
            codec: GatewayCodec,
        };
        let mut public = Router::new()
            .route(&routes.query, post(query))
            .route(&routes.chat, post(chat))
            .route(&routes.embed, post(embed))
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
        if let Some(route) = routes.health {
            management = management.route(&route, get(health));
        }
        if let Some(route) = routes.readiness {
            management = management.route(&route, get(readiness));
        }
        if let Some(route) = routes.metrics {
            let metrics = metrics.clone();
            management = management.route(
                &route,
                get(move || {
                    let metrics = metrics.clone();
                    async move { metrics.render() }
                }),
            );
        }
        if let Some(route) = routes.index {
            management = management.route(&route, get(index));
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
    semaphore: Option<Arc<Semaphore>>,
    codec: GatewayCodec,
}

async fn query(
    State(state): State<PublicState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    text_handler(state, Operation::Query, headers, body).await
}

async fn chat(State(state): State<PublicState>, headers: HeaderMap, body: Bytes) -> Response<Body> {
    text_handler(state, Operation::Chat, headers, body).await
}

async fn embed(
    State(state): State<PublicState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let request_id = request_id(&headers);
    let authenticated = match state.authenticator.authenticate(&headers) {
        Ok(authenticated) => authenticated,
        Err(error) => return error_response(&state, Operation::Embed, request_id, error),
    };
    state.metrics.request_started(Operation::Embed, request_id);
    let context = request_context(request_id, authenticated.clone());
    let decoded = match state.codec.decode_embed(&body) {
        Ok(decoded) => decoded,
        Err(error) => return error_response(&state, Operation::Embed, request_id, error),
    };
    if let Err(error) = authorize(&context, &decoded.target) {
        return error_response(&state, Operation::Embed, request_id, error);
    }
    let endpoint = match resolve_endpoint(&state, &decoded.target) {
        Ok(endpoint) => endpoint,
        Err(error) => return error_response(&state, Operation::Embed, request_id, error),
    };
    let _permit = match acquire(&state) {
        Ok(permit) => permit,
        Err(error) => return error_response(&state, Operation::Embed, request_id, error),
    };
    let mut request = decoded.request;
    request.endpoint = Some(endpoint);
    let run = state.runtime.client.embed_with_context(
        CogentRequestContext {
            request_id: request_id.map(str::to_string),
        },
        request,
    );
    match run.await {
        Ok(response) => match state.codec.encode_embedding(&decoded.target, &response) {
            Ok(body) => success_response(&state, Operation::Embed, request_id, false, body),
            Err(error) => error_response(&state, Operation::Embed, request_id, error),
        },
        Err(error) => error_response(
            &state,
            Operation::Embed,
            request_id,
            GatewayHttpError::from_gateway_error(error.into()),
        ),
    }
}

async fn text_handler(
    state: PublicState,
    operation: Operation,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let request_id = request_id(&headers);
    let authenticated = match state.authenticator.authenticate(&headers) {
        Ok(authenticated) => authenticated,
        Err(error) => return error_response(&state, operation, request_id, error),
    };
    state.metrics.request_started(operation, request_id);
    let context = request_context(request_id, authenticated.clone());
    match operation {
        Operation::Query => {
            let decoded = match state.codec.decode_query(&body) {
                Ok(decoded) => decoded,
                Err(error) => return error_response(&state, operation, request_id, error),
            };
            if let Err(error) = authorize(&context, &decoded.target) {
                return error_response(&state, operation, request_id, error);
            }
            let endpoint = match resolve_endpoint(&state, &decoded.target) {
                Ok(endpoint) => endpoint,
                Err(error) => return error_response(&state, operation, request_id, error),
            };
            let permit = match acquire(&state) {
                Ok(permit) => permit,
                Err(error) => return error_response(&state, operation, request_id, error),
            };
            let mut request = decoded.request;
            request.endpoint = Some(endpoint);
            request.emit_tokens = decoded.stream;
            let run = state.runtime.client.query_with_context(
                CogentRequestContext {
                    request_id: request_id.map(str::to_string),
                },
                request,
            );
            if decoded.stream {
                stream_response(&state, operation, request_id, run.into_parts(), permit)
            } else {
                let _permit = permit;
                match run.await {
                    Ok(response) => match state.codec.encode_text(&decoded.target, &response) {
                        Ok(body) => success_response(&state, operation, request_id, false, body),
                        Err(error) => error_response(&state, operation, request_id, error),
                    },
                    Err(error) => error_response(
                        &state,
                        operation,
                        request_id,
                        GatewayHttpError::from_gateway_error(error.into()),
                    ),
                }
            }
        }
        Operation::Chat => {
            let decoded = match state.codec.decode_chat(&body) {
                Ok(decoded) => decoded,
                Err(error) => return error_response(&state, operation, request_id, error),
            };
            if let Err(error) = authorize(&context, &decoded.target) {
                return error_response(&state, operation, request_id, error);
            }
            let endpoint = match resolve_endpoint(&state, &decoded.target) {
                Ok(endpoint) => endpoint,
                Err(error) => return error_response(&state, operation, request_id, error),
            };
            let permit = match acquire(&state) {
                Ok(permit) => permit,
                Err(error) => return error_response(&state, operation, request_id, error),
            };
            let mut request = decoded.request;
            request.endpoint = Some(endpoint);
            request.emit_tokens = decoded.stream;
            let run = state.runtime.client.chat_with_context(
                CogentRequestContext {
                    request_id: request_id.map(str::to_string),
                },
                request,
            );
            if decoded.stream {
                stream_response(&state, operation, request_id, run.into_parts(), permit)
            } else {
                let _permit = permit;
                match run.await {
                    Ok(response) => match state.codec.encode_text(&decoded.target, &response) {
                        Ok(body) => success_response(&state, operation, request_id, false, body),
                        Err(error) => error_response(&state, operation, request_id, error),
                    },
                    Err(error) => error_response(
                        &state,
                        operation,
                        request_id,
                        GatewayHttpError::from_gateway_error(error.into()),
                    ),
                }
            }
        }
        Operation::Embed => unreachable!("embed uses its dedicated handler"),
    }
}

fn resolve_endpoint(
    state: &PublicState,
    target: &str,
) -> ToolkitResult<cogentlm_client::EndpointRef> {
    state.runtime.targets.get(target).cloned().ok_or_else(|| {
        GatewayHttpError::new(StatusCode::NOT_FOUND, "resolution", "target not found")
    })
}

fn authorize(
    context: &cogentlm_gateway_core::GatewayRequestContext,
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

fn acquire(state: &PublicState) -> ToolkitResult<Option<OwnedSemaphorePermit>> {
    state
        .semaphore
        .as_ref()
        .map(|semaphore| semaphore.clone().try_acquire_owned())
        .transpose()
        .map_err(|_| {
            GatewayHttpError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "admission",
                "application concurrency limit exceeded",
            )
        })
}

fn stream_response(
    state: &PublicState,
    operation: Operation,
    request_id: Option<&str>,
    run: (CogentTokenBatches, CogentTextResponseFuture),
    permit: Option<OwnedSemaphorePermit>,
) -> Response<Body> {
    let stream = text_event_stream(run, permit).map({
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
    state
        .metrics
        .request_finished(operation, request_id, StatusCode::OK);
    response(
        StatusCode::OK,
        state.codec.content_type(true),
        Body::from_stream(stream),
        request_id,
    )
}

struct TextStreamState {
    tokens: CogentTokenBatches,
    response: Option<CogentTextResponseFuture>,
    pending: VecDeque<ToolkitResult<GatewayStreamEvent>>,
    terminal: bool,
    _permit: Option<OwnedSemaphorePermit>,
}

fn text_event_stream(
    (tokens, response): (CogentTokenBatches, CogentTextResponseFuture),
    permit: Option<OwnedSemaphorePermit>,
) -> impl Stream<Item = ToolkitResult<GatewayStreamEvent>> + Send {
    let state = TextStreamState {
        tokens,
        response: Some(response),
        pending: VecDeque::new(),
        terminal: false,
        _permit: permit,
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
    response: cogentlm_client::CogentResult<cogentlm_client::CogentTextResponse>,
) {
    state.terminal = true;
    state._permit.take();
    match response {
        Ok(response) => {
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
            state
                .pending
                .push_back(Err(GatewayHttpError::from_gateway_error(error.into())));
        }
    }
}

fn success_response(
    state: &PublicState,
    operation: Operation,
    request_id: Option<&str>,
    streaming: bool,
    body: Bytes,
) -> Response<Body> {
    state
        .metrics
        .request_finished(operation, request_id, StatusCode::OK);
    response(
        StatusCode::OK,
        state.codec.content_type(streaming),
        Body::from(body),
        request_id,
    )
}

fn error_response(
    state: &PublicState,
    operation: Operation,
    request_id: Option<&str>,
    error: GatewayHttpError,
) -> Response<Body> {
    state
        .metrics
        .request_finished(operation, request_id, error.status);
    let body = state.codec.encode_error(&error);
    response(
        error.status,
        state.codec.content_type(false),
        Body::from(body),
        request_id,
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
