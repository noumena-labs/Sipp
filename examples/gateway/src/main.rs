use std::{
    convert::Infallible,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use anyhow::Context;
use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER},
        HeaderMap, HeaderName, HeaderValue, Method, StatusCode,
    },
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use cogentlm_gateway::{
    constant_time_eq, finish_reason, validate_gateway_bearer_secret, BackendEmbeddingOutput,
    BackendTextOutput, ChatRequestBody, EmbedRequestBody, EmbeddingResponseBody, GatewayAccess,
    GatewayAdapter, GatewayCaller, GatewayError, GatewayErrorKind, GatewayFileConfig,
    GatewayResult, GatewayStreamEvent, QueryRequestBody, TextResponseBody, UsageBody,
};
use futures_util::{Stream, StreamExt};
use serde_json::json;
use tower_http::cors::CorsLayer;

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
const RETRY_AFTER_MS: HeaderName = HeaderName::from_static("retry-after-ms");

#[derive(Debug, Parser)]
#[command(name = "cogentlm-gateway-example")]
#[command(about = "Run a minimal CogentLM gateway example proxy")]
struct Cli {
    /// Path to a gateway TOML configuration.
    #[arg(long)]
    config: PathBuf,
}

// This is the entire application state the minimal proxy needs. Production
// concerns such as request history and admin access live in `apps/gateway-server`.
struct AppState {
    adapter: GatewayAdapter,
    token: String,
    access: GatewayAccess,
    next_request_id: AtomicU64,
}

impl AppState {
    fn authorize(&self, headers: &HeaderMap) -> GatewayResult<GatewayCaller> {
        let token = bearer_token(headers)?;
        if constant_time_eq(token.as_bytes(), self.token.as_bytes()) {
            return Ok(GatewayCaller {
                id: Some("example-token".to_string()),
                access: self.access.clone(),
            });
        }
        Err(GatewayError::new(
            GatewayErrorKind::Authentication,
            "invalid bearer token",
        ))
    }

    fn request_id(&self) -> String {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        format!("gw_example_{id:016x}")
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // The TOML config is shared with the full gateway server, but this example
    // only uses the pieces required to expose a minimal HTTP proxy.
    let config = GatewayFileConfig::from_path(&cli.config)
        .with_context(|| format!("failed to load gateway config {}", cli.config.display()))?;
    let bind = config.server.bind;
    let access = config
        .gateway_access()
        .context("failed to build gateway token access")?;
    let token = required_env(&config.auth.token_env)?;
    validate_gateway_bearer_secret(&token, &config.auth.token_env)?;
    let max_request_bytes = config
        .limits
        .max_request_bytes()
        .context("invalid gateway request byte limit")?;
    let allowed_origins = cors_origins(&config.cors.allowed_origins)?;

    // `GatewayAdapter` is the framework-neutral object from `crates/gateway`.
    // The Axum routes below are intentionally local to this learning example.
    let adapter = config
        .build_adapter()
        .await
        .context("failed to build gateway adapter")?;
    let state = Arc::new(AppState {
        adapter,
        token,
        access,
        next_request_id: AtomicU64::new(1),
    });

    let app = apply_cors(
        Router::new()
            .route("/", get(index))
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .route("/app-healthz", get(healthz))
            .route("/v1/query", post(query))
            .route("/v1/chat", post(chat))
            .route("/v1/embed", post(embed))
            .with_state(state)
            .layer(DefaultBodyLimit::max(max_request_bytes)),
        allowed_origins,
    );
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind gateway example to {bind}"))?;

    println!("minimal gateway example listening on {bind}");
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("gateway example server stopped with an error")
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn readyz() -> &'static str {
    "ready"
}

async fn query(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<QueryRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    let caller = state.authorize(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let request_id = state.request_id();
    let model = body.model.clone();
    let stream = body.stream;
    let request = body.into_backend();

    // Gateway text operations support either one JSON response or a stream of
    // SSE events. The adapter owns backend execution; this route only chooses
    // the HTTP shape.
    if stream {
        let stream = state.adapter.stream_query(&caller, &model, request).await?;
        return Ok(with_request_id(&request_id, sse(stream)));
    }

    let output = state.adapter.query(&caller, &model, request).await?;
    Ok(with_request_id(
        &request_id,
        Json(text_response(&request_id, model, output)),
    ))
}

async fn chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<ChatRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    let caller = state.authorize(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let request_id = state.request_id();
    let model = body.model.clone();
    let stream = body.stream;
    let request = body.into_backend();

    // Chat uses the same transport split as query: JSON for finite responses,
    // SSE for token streaming.
    if stream {
        let stream = state.adapter.stream_chat(&caller, &model, request).await?;
        return Ok(with_request_id(&request_id, sse(stream)));
    }

    let output = state.adapter.chat(&caller, &model, request).await?;
    Ok(with_request_id(
        &request_id,
        Json(text_response(&request_id, model, output)),
    ))
}

async fn embed(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<EmbedRequestBody>, JsonRejection>,
) -> Result<Response, GatewayHttpError> {
    let caller = state.authorize(&headers)?;
    let Json(body) = body.map_err(json_rejection_error)?;
    let request_id = state.request_id();
    let model = body.model.clone();
    let request = body.into_backend();
    let output = state.adapter.embed(&caller, &model, request).await?;
    Ok(with_request_id(
        &request_id,
        Json(embedding_response(&request_id, model, output)),
    ))
}

// The public protocol structs live in `crates/gateway`; the example only fills
// in IDs and converts backend outputs into those wire bodies.
fn text_response(request_id: &str, model: String, output: BackendTextOutput) -> TextResponseBody {
    TextResponseBody {
        id: request_id.to_string(),
        model,
        text: output.text,
        finish_reason: finish_reason(output.finish_reason),
        usage: output.usage.and_then(usage_body),
    }
}

fn embedding_response(
    request_id: &str,
    model: String,
    output: BackendEmbeddingOutput,
) -> EmbeddingResponseBody {
    EmbeddingResponseBody {
        id: request_id.to_string(),
        model,
        embedding: output.values,
        usage: output.usage.and_then(usage_body),
    }
}

fn usage_body(usage: impl Into<UsageBody>) -> Option<UsageBody> {
    let usage = usage.into();
    if usage.input_tokens.is_none() && usage.output_tokens.is_none() && usage.total_tokens.is_none()
    {
        None
    } else {
        Some(usage)
    }
}

fn sse(
    stream: impl Stream<Item = GatewayResult<GatewayStreamEvent>> + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Sse::new(stream.filter_map(|event| async move { sse_event(event).map(Ok) }))
        .keep_alive(KeepAlive::default())
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

fn with_request_id(request_id: &str, response: impl IntoResponse) -> Response {
    let mut response = response.into_response();
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert(X_REQUEST_ID.clone(), value);
    }
    response
}

// This local error wrapper is the minimal HTTP mapping around `GatewayError`.
// The full server adds logging, history, admin APIs, and richer observability.
#[derive(Debug)]
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
        insert_header_if_valid(&mut headers, RETRY_AFTER, retry_after.as_secs().to_string());
        insert_header_if_valid(
            &mut headers,
            RETRY_AFTER_MS.clone(),
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

fn bearer_token(headers: &HeaderMap) -> GatewayResult<&str> {
    let Some(header) = headers.get(AUTHORIZATION) else {
        return Err(GatewayError::new(
            GatewayErrorKind::Authentication,
            "missing gateway bearer token",
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

fn apply_cors(router: Router, allowed_origins: Vec<HeaderValue>) -> Router {
    if allowed_origins.is_empty() {
        return router;
    }
    router.layer(
        CorsLayer::new()
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([AUTHORIZATION, CONTENT_TYPE])
            .expose_headers([X_REQUEST_ID.clone(), RETRY_AFTER, RETRY_AFTER_MS.clone()])
            .allow_origin(allowed_origins),
    )
}

fn cors_origins(origins: &[String]) -> GatewayResult<Vec<HeaderValue>> {
    let mut headers = Vec::with_capacity(origins.len());
    for origin in origins {
        let trimmed = origin.trim();
        if trimmed.is_empty() || trimmed != origin {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "CORS origins must be exact non-empty origins",
            ));
        }
        let value = HeaderValue::from_str(origin).map_err(|error| {
            GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("invalid CORS origin {origin}: {error}"),
            )
        })?;
        headers.push(value);
    }
    Ok(headers)
}

fn required_env(name: &str) -> anyhow::Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
    if value.trim().is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    Ok(value)
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

// Keeping the root page inline makes it clear that this page comes from the
// learning example, not from the production gateway dashboard.
const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CogentLM Minimal Gateway Example</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; max-width: 760px; }
    code { background: #f3f4f6; padding: .1rem .25rem; }
    li { margin-block: .35rem; }
  </style>
</head>
<body>
  <h1>CogentLM Minimal Gateway Example</h1>
  <p>This page is served by <code>examples/gateway/src/main.rs</code>. It is a small learning proxy built directly from <code>crates/gateway</code>, not the production gateway dashboard.</p>
  <p>The production-style proxy lives in <code>apps/gateway-server</code> and adds admin status, request history, and dashboard features.</p>
  <ul>
    <li><code>GET /healthz</code> and <code>GET /readyz</code>: basic probes.</li>
    <li><code>GET /app-healthz</code>: example app route beside the gateway routes.</li>
    <li><code>POST /v1/query</code>: raw prompt text generation.</li>
    <li><code>POST /v1/chat</code>: chat message generation.</li>
    <li><code>POST /v1/embed</code>: text embeddings.</li>
  </ul>
</body>
</html>
"#;
