use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::core::{ChatRole, FinishReason, TokenBatch, TokenEmissionStats, TokenUsage};
use futures::StreamExt;
use futures_channel::mpsc;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};

use crate::client::dispatch::InferenceEndpoint;
use crate::client::gateway::{GatewayAuthentication, GatewayEndpointConfig, GatewaySecret};
use crate::client::io_executor::IoExecutor;
use crate::client::{
    validate, SippChatRequest, SippEmbedRequest, SippEmbeddingResponse, SippEmbeddingRun,
    SippError, SippQueryRequest, SippRequestContext, SippResponseMetadata, SippResult,
    SippTextResponse, SippTextRun, SippTokenBatches, EndpointCapabilities, EndpointError,
    EndpointRef,
};

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) struct GatewayEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
    target: String,
    routes: crate::client::GatewayRoutes,
    protocol_options: serde_json::Map<String, serde_json::Value>,
    transport: GatewayTransport,
    executor: IoExecutor,
}

impl GatewayEndpoint {
    pub(crate) fn new(
        endpoint: EndpointRef,
        config: GatewayEndpointConfig,
        executor: IoExecutor,
    ) -> SippResult<Self> {
        validate_name(&config.target, "gateway target")?;
        validate_routes(&config.routes)?;
        let transport = GatewayTransport::new(
            config.base_url,
            config.authentication,
            config.static_headers,
            config.timeouts,
        )?;
        Ok(Self {
            endpoint,
            capabilities: EndpointCapabilities::unknown(),
            target: config.target,
            routes: config.routes,
            protocol_options: config.protocol_options,
            transport,
            executor,
        })
    }
}

impl InferenceEndpoint for GatewayEndpoint {
    fn endpoint(&self) -> &EndpointRef {
        &self.endpoint
    }

    fn capabilities(&self) -> &EndpointCapabilities {
        &self.capabilities
    }

    fn query_with_context(
        &self,
        context: SippRequestContext,
        request: SippQueryRequest,
    ) -> SippTextRun {
        if let Err(error) = validate::gateway_query(&request) {
            return SippTextRun::ready_err(error);
        }
        let body = match query_value(
            &self.target,
            request.clone(),
            request.emit_tokens,
            &self.protocol_options,
        ) {
            Ok(body) => body,
            Err(error) => return SippTextRun::ready_err(error),
        };
        self.text_run(
            context,
            self.routes.query.clone(),
            body,
            request.emit_tokens,
        )
    }

    fn chat_with_context(
        &self,
        context: SippRequestContext,
        request: SippChatRequest,
    ) -> SippTextRun {
        if let Err(error) = validate::gateway_chat(&request) {
            return SippTextRun::ready_err(error);
        }
        let body = match chat_value(
            &self.target,
            request.clone(),
            request.emit_tokens,
            &self.protocol_options,
        ) {
            Ok(body) => body,
            Err(error) => return SippTextRun::ready_err(error),
        };
        self.text_run(context, self.routes.chat.clone(), body, request.emit_tokens)
    }

    fn embed_with_context(
        &self,
        context: SippRequestContext,
        request: SippEmbedRequest,
    ) -> SippEmbeddingRun {
        if let Err(error) = validate::gateway_embed(&request) {
            return SippEmbeddingRun::ready_err(error);
        }
        let body = match embed_value(&self.target, request, &self.protocol_options) {
            Ok(body) => body,
            Err(error) => return SippEmbeddingRun::ready_err(error),
        };
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let route = self.routes.embed.clone();
        let request_id = context.request_id;
        let executor = self.executor.clone();
        let join = executor.spawn(async move {
            let response = transport.post_json(&route, body).await?;
            decode_embedding_response(endpoint, request_id, response.body)
        });
        SippEmbeddingRun::new(Box::pin(GatewayResponseFuture::new(join, executor)))
    }
}

impl GatewayEndpoint {
    fn text_run(
        &self,
        context: SippRequestContext,
        route: String,
        body: serde_json::Value,
        stream: bool,
    ) -> SippTextRun {
        let transport = self.transport.clone();
        let endpoint = self.endpoint.clone();
        let request_id = context.request_id;
        let executor = self.executor.clone();
        if stream {
            let (batch_tx, batch_rx) = mpsc::unbounded();
            let join = executor.spawn(async move {
                transport
                    .post_stream(&route, body, endpoint, request_id, batch_tx)
                    .await
            });
            SippTextRun::new(
                Box::pin(GatewayResponseFuture::new(join, executor)),
                SippTokenBatches::from_receiver(batch_rx),
            )
        } else {
            let join = executor.spawn(async move {
                let response = transport.post_json(&route, body).await?;
                decode_text_response(endpoint, request_id, response.body)
            });
            SippTextRun::new(
                Box::pin(GatewayResponseFuture::new(join, executor)),
                SippTokenBatches::closed(),
            )
        }
    }
}

#[derive(Clone)]
struct GatewayTransport {
    client: reqwest::Client,
    base_url: String,
    headers: HeaderMap,
    request_timeout: std::time::Duration,
    secrets: Arc<Vec<String>>,
}

struct GatewayResponse {
    body: serde_json::Value,
}

impl GatewayTransport {
    fn new(
        base_url: String,
        authentication: GatewayAuthentication,
        static_headers: std::collections::BTreeMap<String, String>,
        timeouts: crate::client::GatewayTimeoutPolicy,
    ) -> SippResult<Self> {
        let base_url = base_url.trim_end_matches('/').to_string();
        let parsed = reqwest::Url::parse(&base_url)
            .map_err(|error| endpoint_error("configuration", error.to_string()))?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err(endpoint_error(
                "configuration",
                "base_url must be an absolute HTTP(S) URL",
            ));
        }
        if timeouts.connect.is_zero() || timeouts.request.is_zero() || timeouts.read.is_zero() {
            return Err(endpoint_error(
                "configuration",
                "timeouts must be greater than zero",
            ));
        }

        let mut headers = HeaderMap::new();
        let mut secrets = Vec::new();
        for (name, value) in static_headers {
            insert_header(&mut headers, &name, &value, false)?;
        }
        match authentication {
            GatewayAuthentication::None => {}
            GatewayAuthentication::Bearer(secret) => {
                let value = format!("Bearer {}", secret.expose());
                insert_header(&mut headers, AUTHORIZATION.as_str(), &value, true)?;
                secrets.push(secret.expose().to_string());
            }
            GatewayAuthentication::Header { name, value } => {
                insert_header(&mut headers, &name, value.expose(), true)?;
                secrets.push(value.expose().to_string());
            }
        }

        let client = reqwest::Client::builder()
            .connect_timeout(timeouts.connect)
            .read_timeout(timeouts.read)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| endpoint_error("transport", error.to_string()))?;
        Ok(Self {
            client,
            base_url,
            headers,
            request_timeout: timeouts.request,
            secrets: Arc::new(secrets),
        })
    }

    async fn post_json(
        &self,
        route: &str,
        body: serde_json::Value,
    ) -> SippResult<GatewayResponse> {
        let response = self
            .client
            .post(self.url(route))
            .headers(self.headers.clone())
            .timeout(self.request_timeout)
            .json(&body)
            .send()
            .await
            .map_err(|error| self.transport_error(error))?;
        if !response.status().is_success() {
            return Err(self.response_error(response).await);
        }
        let body = response
            .json()
            .await
            .map_err(|error| self.transport_error(error))?;
        Ok(GatewayResponse { body })
    }

    async fn post_stream(
        &self,
        route: &str,
        body: serde_json::Value,
        endpoint: EndpointRef,
        request_id: Option<String>,
        sender: mpsc::UnboundedSender<TokenBatch>,
    ) -> SippResult<SippTextResponse> {
        let response = self
            .client
            .post(self.url(route))
            .headers(self.headers.clone())
            .json(&body)
            .send()
            .await
            .map_err(|error| self.transport_error(error))?;
        if !response.status().is_success() {
            return Err(self.response_error(response).await);
        }

        let mut bytes = response.bytes_stream();
        let mut parser = SseParser::default();
        let mut text = String::new();
        let mut usage = None;
        let mut finish_reason = None;
        let mut sequence = 0;
        let mut stats = TokenEmissionStats::default();
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(|error| self.transport_error(error))?;
            for event in parser.push(&chunk)? {
                apply_event(
                    event,
                    &request_id,
                    &sender,
                    &mut text,
                    &mut usage,
                    &mut finish_reason,
                    &mut sequence,
                    &mut stats,
                )?;
            }
        }
        for event in parser.finish()? {
            apply_event(
                event,
                &request_id,
                &sender,
                &mut text,
                &mut usage,
                &mut finish_reason,
                &mut sequence,
                &mut stats,
            )?;
        }
        let finish_reason = finish_reason
            .ok_or_else(|| endpoint_error("protocol", "stream ended before a done event"))?;
        Ok(SippTextResponse {
            endpoint,
            text,
            finish_reason,
            usage,
            local_stats: None,
            metadata: SippResponseMetadata {
                request_id,
                upstream_request_id: None,
                upstream_response_id: None,
            },
        })
    }

    fn url(&self, route: &str) -> String {
        format!("{}/{}", self.base_url, route.trim_start_matches('/'))
    }

    async fn response_error(&self, response: reqwest::Response) -> SippError {
        let status = response.status().as_u16();
        let raw = response
            .json::<serde_json::Value>()
            .await
            .unwrap_or(serde_json::Value::Null);
        let message = raw
            .pointer("/error/message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("gateway endpoint request failed");
        let mut error = EndpointError::new("http", redact(message, &self.secrets));
        error.status = Some(status);
        error.code = raw
            .pointer("/error/code")
            .and_then(serde_json::Value::as_str)
            .map(|value| redact(value, &self.secrets));
        error.raw = Some(Box::new(redact_value(raw, &self.secrets)));
        SippError::Endpoint(error)
    }

    fn transport_error(&self, error: reqwest::Error) -> SippError {
        endpoint_error(
            if error.is_timeout() {
                "timeout"
            } else {
                "transport"
            },
            redact(&error.to_string(), &self.secrets),
        )
    }
}

struct GatewayResponseFuture<T> {
    join: tokio::task::JoinHandle<SippResult<T>>,
    _executor: IoExecutor,
}

impl<T> GatewayResponseFuture<T> {
    fn new(join: tokio::task::JoinHandle<SippResult<T>>, executor: IoExecutor) -> Self {
        Self {
            join,
            _executor: executor,
        }
    }
}

impl<T> Future for GatewayResponseFuture<T> {
    type Output = SippResult<T>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.join).poll(context) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(error)) => Poll::Ready(Err(SippError::Internal(format!(
                "gateway task failed: {error}"
            )))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> Drop for GatewayResponseFuture<T> {
    fn drop(&mut self) {
        self.join.abort();
    }
}

#[derive(Default)]
struct SseParser {
    buffer: Vec<u8>,
}

struct SseEvent {
    name: String,
    data: String,
}

impl SseParser {
    fn push(&mut self, bytes: &[u8]) -> SippResult<Vec<SseEvent>> {
        self.buffer.extend_from_slice(bytes);
        self.take_events(false)
    }

    fn finish(&mut self) -> SippResult<Vec<SseEvent>> {
        self.take_events(true)
    }

    fn take_events(&mut self, finish: bool) -> SippResult<Vec<SseEvent>> {
        const MAX_BUFFER: usize = 1 << 20;
        if self.buffer.len() > MAX_BUFFER {
            return Err(endpoint_error(
                "protocol",
                "SSE event exceeded buffer limit",
            ));
        }
        let mut events = Vec::new();
        loop {
            let Some((index, length)) = find_boundary(&self.buffer) else {
                break;
            };
            let raw = self.buffer.drain(..index + length).collect::<Vec<_>>();
            if let Some(event) = parse_event(&raw[..index])? {
                events.push(event);
            }
        }
        if finish && !self.buffer.is_empty() {
            let raw = std::mem::take(&mut self.buffer);
            if let Some(event) = parse_event(&raw)? {
                events.push(event);
            }
        }
        Ok(events)
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_event(
    event: SseEvent,
    request_id: &Option<String>,
    sender: &mpsc::UnboundedSender<TokenBatch>,
    text: &mut String,
    usage: &mut Option<TokenUsage>,
    finish_reason: &mut Option<FinishReason>,
    sequence: &mut u32,
    stats: &mut TokenEmissionStats,
) -> SippResult<()> {
    let value: serde_json::Value = serde_json::from_str(&event.data)
        .map_err(|error| endpoint_error("protocol", error.to_string()))?;
    match event.name.as_str() {
        "token" => {
            let token = value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| endpoint_error("protocol", "token event missing text"))?;
            text.push_str(token);
            stats.frames_sent += 1;
            stats.batches_sent += 1;
            stats.bytes_sent += token.len() as u64;
            let batch = TokenBatch {
                request_id: request_id.clone().unwrap_or_default(),
                stream_id: 0,
                sequence_start: *sequence,
                text: token.to_string(),
                frame_count: 1,
                byte_count: token.len() as u32,
                stats: *stats,
            };
            *sequence = sequence.wrapping_add(1);
            let _ = sender.unbounded_send(batch);
        }
        "usage" => {
            *usage = Some(decode_usage(value)?);
        }
        "done" => {
            *finish_reason = Some(
                match value
                    .get("finish_reason")
                    .and_then(serde_json::Value::as_str)
                {
                    Some("length") => FinishReason::Length,
                    _ => FinishReason::Stop,
                },
            );
        }
        "error" => {
            let message = value
                .pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("stream error");
            return Err(endpoint_error("protocol", message));
        }
        name => {
            return Err(endpoint_error(
                "protocol",
                format!("unsupported SSE event: {name}"),
            ));
        }
    }
    Ok(())
}

fn find_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| (index, 4))
        })
}

fn parse_event(raw: &[u8]) -> SippResult<Option<SseEvent>> {
    let raw =
        std::str::from_utf8(raw).map_err(|error| endpoint_error("protocol", error.to_string()))?;
    let mut name = "message".to_string();
    let mut data = Vec::new();
    for line in raw.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            name = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.trim_start());
        }
    }
    if data.is_empty() {
        Ok(None)
    } else {
        Ok(Some(SseEvent {
            name,
            data: data.join("\n"),
        }))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct QueryBody {
    model: String,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    #[serde(default)]
    stop: Vec<String>,
    #[serde(default)]
    stream: bool,
    #[serde(flatten)]
    options: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatBody {
    model: String,
    messages: Vec<ChatMessageBody>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    #[serde(default)]
    stop: Vec<String>,
    #[serde(default)]
    stream: bool,
    #[serde(flatten)]
    options: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatMessageBody {
    role: ChatRole,
    content: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct EmbedBody {
    model: String,
    input: String,
    #[serde(flatten)]
    options: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct UsageBody {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl From<TokenUsage> for UsageBody {
    fn from(usage: TokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl From<UsageBody> for TokenUsage {
    fn from(usage: UsageBody) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

fn query_value(
    target: &str,
    request: SippQueryRequest,
    stream: bool,
    protocol_options: &serde_json::Map<String, serde_json::Value>,
) -> SippResult<serde_json::Value> {
    let mut options = protocol_options.clone();
    options.extend(request.endpoint_options);
    serde_json::to_value(QueryBody {
        model: target.to_string(),
        prompt: request.prompt,
        max_tokens: request.options.max_tokens,
        temperature: request.options.temperature,
        top_p: request.options.top_p,
        stop: request.options.stop,
        stream,
        options,
    })
    .map_err(|error| endpoint_error("protocol", error.to_string()))
}

fn chat_value(
    target: &str,
    request: SippChatRequest,
    stream: bool,
    protocol_options: &serde_json::Map<String, serde_json::Value>,
) -> SippResult<serde_json::Value> {
    let mut options = protocol_options.clone();
    options.extend(request.endpoint_options);
    serde_json::to_value(ChatBody {
        model: target.to_string(),
        messages: request
            .messages
            .into_iter()
            .map(|message| ChatMessageBody {
                role: message.role,
                content: message.content,
            })
            .collect(),
        max_tokens: request.options.max_tokens,
        temperature: request.options.temperature,
        top_p: request.options.top_p,
        stop: request.options.stop,
        stream,
        options,
    })
    .map_err(|error| endpoint_error("protocol", error.to_string()))
}

fn embed_value(
    target: &str,
    request: SippEmbedRequest,
    protocol_options: &serde_json::Map<String, serde_json::Value>,
) -> SippResult<serde_json::Value> {
    let mut options = protocol_options.clone();
    options.extend(request.endpoint_options);
    serde_json::to_value(EmbedBody {
        model: target.to_string(),
        input: request.input,
        options,
    })
    .map_err(|error| endpoint_error("protocol", error.to_string()))
}

fn decode_text_response(
    endpoint: EndpointRef,
    request_id: Option<String>,
    body: serde_json::Value,
) -> SippResult<SippTextResponse> {
    let text = required_string(&body, "text")?;
    let finish_reason = match required_string(&body, "finish_reason")?.as_str() {
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    };
    Ok(SippTextResponse {
        endpoint,
        text,
        finish_reason,
        usage: optional_usage(&body)?,
        local_stats: None,
        metadata: SippResponseMetadata {
            request_id,
            upstream_request_id: None,
            upstream_response_id: body
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        },
    })
}

fn decode_embedding_response(
    endpoint: EndpointRef,
    request_id: Option<String>,
    body: serde_json::Value,
) -> SippResult<SippEmbeddingResponse> {
    let values = body
        .get("embedding")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| endpoint_error("protocol", "embedding response missing vector"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .map(|value| value as f32)
                .ok_or_else(|| endpoint_error("protocol", "embedding value is not numeric"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SippEmbeddingResponse {
        endpoint,
        values,
        usage: optional_usage(&body)?,
        local_stats: None,
        pooling: None,
        normalized: None,
        metadata: SippResponseMetadata {
            request_id,
            upstream_request_id: None,
            upstream_response_id: body
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        },
    })
}

fn required_string(body: &serde_json::Value, field: &'static str) -> SippResult<String> {
    body.get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| endpoint_error("protocol", format!("response missing {field}")))
}

fn optional_usage(body: &serde_json::Value) -> SippResult<Option<TokenUsage>> {
    let Some(usage) = body.get("usage").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    serde_json::from_value::<UsageBody>(usage.clone())
        .map(|usage| Some(usage.into()))
        .map_err(|error| endpoint_error("protocol", error.to_string()))
}

fn decode_usage(value: serde_json::Value) -> SippResult<TokenUsage> {
    serde_json::from_value::<UsageBody>(value)
        .map(TokenUsage::from)
        .map_err(|error| endpoint_error("protocol", error.to_string()))
}

fn insert_header(
    headers: &mut HeaderMap,
    name: &str,
    value: &str,
    sensitive: bool,
) -> SippResult<()> {
    let name = HeaderName::from_bytes(name.as_bytes())
        .map_err(|error| endpoint_error("configuration", error.to_string()))?;
    let mut value = HeaderValue::from_str(value)
        .map_err(|error| endpoint_error("configuration", error.to_string()))?;
    value.set_sensitive(sensitive);
    headers.insert(name, value);
    Ok(())
}

fn validate_name(value: &str, field: &str) -> SippResult<()> {
    if value.is_empty() || value.trim() != value {
        Err(endpoint_error(
            "configuration",
            format!("{field} must be a non-empty trimmed value"),
        ))
    } else {
        Ok(())
    }
}

fn validate_routes(routes: &crate::client::GatewayRoutes) -> SippResult<()> {
    for (name, route) in [
        ("query route", routes.query.as_str()),
        ("chat route", routes.chat.as_str()),
        ("embed route", routes.embed.as_str()),
    ] {
        if !route.starts_with('/') || route.contains('?') || route.contains('#') {
            return Err(endpoint_error(
                "configuration",
                format!("{name} must be an absolute path"),
            ));
        }
    }
    Ok(())
}

fn endpoint_error(kind: impl Into<String>, message: impl Into<String>) -> SippError {
    SippError::Endpoint(EndpointError::new(kind, message))
}

fn redact(value: &str, secrets: &[String]) -> String {
    secrets
        .iter()
        .filter(|secret| !secret.is_empty())
        .fold(value.to_string(), |value, secret| {
            value.replace(secret, "[redacted]")
        })
}

fn redact_value(value: serde_json::Value, secrets: &[String]) -> serde_json::Value {
    match value {
        serde_json::Value::String(value) => serde_json::Value::String(redact(&value, secrets)),
        serde_json::Value::Array(values) => serde_json::Value::Array(
            values
                .into_iter()
                .map(|value| redact_value(value, secrets))
                .collect(),
        ),
        serde_json::Value::Object(values) => serde_json::Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, redact_value(value, secrets)))
                .collect(),
        ),
        value => value,
    }
}

#[allow(dead_code)]
fn _secret_type_is_redacted(_: GatewaySecret) {}
