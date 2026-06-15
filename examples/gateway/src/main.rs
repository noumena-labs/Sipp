use std::collections::VecDeque;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Context;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode};
use axum::response::Html;
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use clap::Parser;
use futures_util::future::{select, Either};
use futures_util::{stream, Stream, StreamExt};
use sipp::engine::{NativeRuntimeConfig, PoolingType};
use sipp::gateway_core::{GatewayStreamEvent, Operation};
use sipp::{
    EndpointDescriptor, EndpointRef, SippClient, SippRequestContext, SippTextResponseFuture,
    SippTokenBatches,
};
use sipp_gateway::{request_id, GatewayCodec, GatewayHttpError, ProtocolCodec};

const INDEX_HTML: &str = include_str!("../assets/index.html");

#[derive(Debug, Parser)]
#[command(name = "sipp-gateway-example")]
#[command(about = "Run an explicit Axum gateway route example")]
struct Cli {
    /// Local GGUF model to load.
    #[arg(long)]
    model: PathBuf,
    /// Listener address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut client = SippClient::new();
    let text_endpoint = client
        .add(
            "local-text",
            EndpointDescriptor::local(cli.model.clone(), NativeRuntimeConfig::default()),
        )
        .await
        .context("failed to load local text model")?;
    let mut embedding_runtime = NativeRuntimeConfig::default();
    embedding_runtime.context.embeddings = Some(true);
    embedding_runtime.context.pooling = Some(PoolingType::Mean);
    let embedding_endpoint = client
        .add(
            "local-embed",
            EndpointDescriptor::local(cli.model, embedding_runtime),
        )
        .await
        .context("failed to load local embedding model")?;

    let state = AppState {
        client: Arc::new(client),
        text_endpoint,
        embedding_endpoint,
        next_request_id: Arc::new(AtomicU64::new(1)),
        codec: GatewayCodec,
    };
    let app = Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
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

#[derive(Clone)]
struct AppState {
    client: Arc<SippClient>,
    text_endpoint: EndpointRef,
    embedding_endpoint: EndpointRef,
    next_request_id: Arc<AtomicU64>,
    codec: GatewayCodec,
}

async fn query(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response<Body> {
    let request_id = request_id(&headers)
        .map(str::to_string)
        .unwrap_or_else(|| state.next_request_id());
    let decoded = match state.codec.decode_query(&body) {
        Ok(decoded) => decoded,
        Err(error) => return error_response(&state, Some(&request_id), error),
    };
    let endpoint = match state.resolve(&decoded.target, Operation::Query) {
        Ok(endpoint) => endpoint,
        Err(error) => return error_response(&state, Some(&request_id), error),
    };
    let mut request = decoded.request;
    request.endpoint = Some(endpoint);
    request.emit_tokens = decoded.stream;
    let run = state.client.query_with_context(
        SippRequestContext {
            request_id: Some(request_id.clone()),
        },
        request,
    );
    if decoded.stream {
        stream_response(&state, Some(&request_id), run.into_parts())
    } else {
        match run.await {
            Ok(response) => match state.codec.encode_text(&decoded.target, &response) {
                Ok(body) => success_response(&state, Some(&request_id), false, body),
                Err(error) => error_response(&state, Some(&request_id), error),
            },
            Err(error) => error_response(
                &state,
                Some(&request_id),
                GatewayHttpError::from_gateway_error(error.into()),
            ),
        }
    }
}

async fn chat(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response<Body> {
    let request_id = request_id(&headers)
        .map(str::to_string)
        .unwrap_or_else(|| state.next_request_id());
    let decoded = match state.codec.decode_chat(&body) {
        Ok(decoded) => decoded,
        Err(error) => return error_response(&state, Some(&request_id), error),
    };
    let endpoint = match state.resolve(&decoded.target, Operation::Chat) {
        Ok(endpoint) => endpoint,
        Err(error) => return error_response(&state, Some(&request_id), error),
    };
    let mut request = decoded.request;
    request.endpoint = Some(endpoint);
    request.emit_tokens = decoded.stream;
    let run = state.client.chat_with_context(
        SippRequestContext {
            request_id: Some(request_id.clone()),
        },
        request,
    );
    if decoded.stream {
        stream_response(&state, Some(&request_id), run.into_parts())
    } else {
        match run.await {
            Ok(response) => match state.codec.encode_text(&decoded.target, &response) {
                Ok(body) => success_response(&state, Some(&request_id), false, body),
                Err(error) => error_response(&state, Some(&request_id), error),
            },
            Err(error) => error_response(
                &state,
                Some(&request_id),
                GatewayHttpError::from_gateway_error(error.into()),
            ),
        }
    }
}

async fn embed(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response<Body> {
    let request_id = request_id(&headers)
        .map(str::to_string)
        .unwrap_or_else(|| state.next_request_id());
    let decoded = match state.codec.decode_embed(&body) {
        Ok(decoded) => decoded,
        Err(error) => return error_response(&state, Some(&request_id), error),
    };
    let endpoint = match state.resolve(&decoded.target, Operation::Embed) {
        Ok(endpoint) => endpoint,
        Err(error) => return error_response(&state, Some(&request_id), error),
    };
    let mut request = decoded.request;
    request.endpoint = Some(endpoint);
    let run = state.client.embed_with_context(
        SippRequestContext {
            request_id: Some(request_id.clone()),
        },
        request,
    );
    match run.await {
        Ok(response) => match state.codec.encode_embedding(&decoded.target, &response) {
            Ok(body) => success_response(&state, Some(&request_id), false, body),
            Err(error) => error_response(&state, Some(&request_id), error),
        },
        Err(error) => error_response(
            &state,
            Some(&request_id),
            GatewayHttpError::from_gateway_error(error.into()),
        ),
    }
}

impl AppState {
    fn next_request_id(&self) -> String {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        format!("example_{id:016x}")
    }

    fn resolve(&self, target: &str, operation: Operation) -> Result<EndpointRef, GatewayHttpError> {
        if target != "local" {
            return Err(GatewayHttpError::new(
                StatusCode::NOT_FOUND,
                "resolution",
                "target not found",
            ));
        }
        Ok(match operation {
            Operation::Query | Operation::Chat => self.text_endpoint.clone(),
            Operation::Embed => self.embedding_endpoint.clone(),
        })
    }
}

fn stream_response(
    state: &AppState,
    request_id: Option<&str>,
    run: (SippTokenBatches, SippTextResponseFuture),
) -> Response<Body> {
    let stream = text_event_stream(run).map({
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
        request_id,
    )
}

struct TextStreamState {
    tokens: SippTokenBatches,
    response: Option<SippTextResponseFuture>,
    pending: VecDeque<Result<GatewayStreamEvent, GatewayHttpError>>,
    terminal: bool,
}

fn text_event_stream(
    (tokens, response): (SippTokenBatches, SippTextResponseFuture),
) -> impl Stream<Item = Result<GatewayStreamEvent, GatewayHttpError>> + Send {
    let state = TextStreamState {
        tokens,
        response: Some(response),
        pending: VecDeque::new(),
        terminal: false,
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
    state: &AppState,
    request_id: Option<&str>,
    streaming: bool,
    body: Bytes,
) -> Response<Body> {
    response(
        StatusCode::OK,
        state.codec.content_type(streaming),
        Body::from(body),
        request_id,
    )
}

fn error_response(
    state: &AppState,
    request_id: Option<&str>,
    error: GatewayHttpError,
) -> Response<Body> {
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
