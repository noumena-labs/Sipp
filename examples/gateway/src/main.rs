use std::{
    convert::Infallible,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use anyhow::Context;
use axum::{
    extract::State,
    http::{HeaderName, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use cogentlm_client::{CogentClient, EndpointDescriptor};
use cogentlm_engine::engine::NativeRuntimeConfig;
use cogentlm_gateway::{
    finish_reason, ChatRequestBody, CogentClientExecutor, EmbedRequestBody, EmbeddingResponseBody,
    ErrorEnvelope, GatewayAdapter, GatewayAlias, GatewayAliasLimits, GatewayCaller, GatewayError,
    GatewayRequestContext, GatewayStreamEvent, OperationSet, QueryRequestBody, TextResponseBody,
    UsageBody,
};
use futures_util::StreamExt;
use serde_json::json;

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
const INDEX_HTML: &str = include_str!("../assets/index.html");

#[derive(Debug, Parser)]
#[command(name = "cogentlm-gateway-example")]
#[command(about = "Run the minimal CogentLM Axum gateway example")]
struct Cli {
    /// Local GGUF model to load.
    #[arg(long)]
    model: PathBuf,
    /// Listener address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
}

struct AppState {
    adapter: GatewayAdapter,
    next_request_id: AtomicU64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut client = CogentClient::new();
    let mut runtime = NativeRuntimeConfig::default();
    runtime.context.embeddings = Some(true);
    let endpoint = client
        .add("local", EndpointDescriptor::local(cli.model, runtime))
        .await
        .context("failed to load local model")?;
    let adapter = GatewayAdapter::builder(CogentClientExecutor::new(client))
        .alias(GatewayAlias::new(
            "local",
            endpoint,
            OperationSet::all(),
            GatewayAliasLimits::default(),
        )?)?
        .build()?;
    let state = Arc::new(AppState {
        adapter,
        next_request_id: AtomicU64::new(1),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/v1/query", post(query))
        .route("/v1/chat", post(chat))
        .route("/v1/embed", post(embed))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(cli.bind)
        .await
        .with_context(|| format!("failed to bind {}", cli.bind))?;
    println!("gateway example listening on http://{}", cli.bind);
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("gateway example stopped")
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn query(
    State(state): State<Arc<AppState>>,
    Json(body): Json<QueryRequestBody>,
) -> Result<Response, HttpError> {
    let (request_id, context) = request_context(&state)?;
    let alias = body.model.clone();
    if body.stream {
        let stream = state.adapter.stream_query(&context, body)?;
        return Ok(with_request_id(
            &request_id,
            Sse::new(stream.map(sse_event)).keep_alive(KeepAlive::default()),
        ));
    }
    let output = state.adapter.query(&context, body).await?;
    Ok(with_request_id(
        &request_id,
        Json(TextResponseBody {
            id: request_id.clone(),
            model: alias,
            text: output.text,
            finish_reason: finish_reason(output.finish_reason),
            usage: output.usage.map(UsageBody::from),
        }),
    ))
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatRequestBody>,
) -> Result<Response, HttpError> {
    let (request_id, context) = request_context(&state)?;
    let alias = body.model.clone();
    if body.stream {
        let stream = state.adapter.stream_chat(&context, body)?;
        return Ok(with_request_id(
            &request_id,
            Sse::new(stream.map(sse_event)).keep_alive(KeepAlive::default()),
        ));
    }
    let output = state.adapter.chat(&context, body).await?;
    Ok(with_request_id(
        &request_id,
        Json(TextResponseBody {
            id: request_id.clone(),
            model: alias,
            text: output.text,
            finish_reason: finish_reason(output.finish_reason),
            usage: output.usage.map(UsageBody::from),
        }),
    ))
}

async fn embed(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EmbedRequestBody>,
) -> Result<Response, HttpError> {
    let (request_id, context) = request_context(&state)?;
    let alias = body.model.clone();
    let output = state.adapter.embed(&context, body).await?;
    Ok(with_request_id(
        &request_id,
        Json(EmbeddingResponseBody {
            id: request_id.clone(),
            model: alias,
            embedding: output.values,
            usage: output.usage.map(UsageBody::from),
        }),
    ))
}

fn request_context(state: &AppState) -> Result<(String, GatewayRequestContext), GatewayError> {
    let id = state.next_request_id.fetch_add(1, Ordering::Relaxed);
    let request_id = format!("example_{id:016x}");
    let context = GatewayRequestContext::new(&request_id, GatewayCaller::anonymous())?;
    Ok((request_id, context))
}

fn sse_event(event: Result<GatewayStreamEvent, GatewayError>) -> Result<Event, Infallible> {
    let event = match event {
        Ok(GatewayStreamEvent::TokenBatch(batch)) => {
            Event::default().event("token").json_data(json!({
                "text": batch.text,
                "sequence": batch.sequence_start,
            }))
        }
        Ok(GatewayStreamEvent::Usage { usage }) => Event::default()
            .event("usage")
            .json_data(UsageBody::from(usage)),
        Ok(GatewayStreamEvent::Finished { finish_reason, .. }) => Event::default()
            .event("done")
            .json_data(json!({ "finish_reason": finish_reason.as_str() })),
        Err(error) => Event::default()
            .event("error")
            .json_data(ErrorEnvelope::from(&error)),
    }
    .unwrap_or_else(|_| {
        Event::default()
            .event("error")
            .data(r#"{"error":{"code":"internal","message":"SSE encoding failed"}}"#)
    });
    Ok(event)
}

fn with_request_id(request_id: &str, response: impl IntoResponse) -> Response {
    let mut response = response.into_response();
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert(X_REQUEST_ID.clone(), value);
    }
    response
}

struct HttpError(GatewayError);

impl From<GatewayError> for HttpError {
    fn from(error: GatewayError) -> Self {
        Self(error)
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.0.kind.http_status_code())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(ErrorEnvelope::from(&self.0))).into_response()
    }
}
