use std::net::IpAddr;
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};

use crate::error::gateway_error_kind_from_code;
use crate::request::{chat_body, embed_body, query_body};
use crate::response::{embedding_response_from_body, text_response_from_body};
use crate::stream::{gateway_stream_events, GatewayByteStream};
use crate::{
    GatewayChatRequest, GatewayConfig, GatewayEmbedRequest, GatewayEmbeddingResponse, GatewayError,
    GatewayErrorKind, GatewayQueryRequest, GatewayResult, GatewaySecret, GatewayStream,
    GatewayStreamEvent, GatewayTextResponse,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// HTTP transport for the CogentLM Remote Gateway Protocol.
#[derive(Clone)]
pub struct GatewayTransport {
    http: reqwest::Client,
    base_url: String,
    headers: HeaderMap,
    token: GatewaySecret,
    timeout: Duration,
}

struct HttpResponse {
    request_id: Option<String>,
    body: serde_json::Value,
}

struct HttpStreamResponse {
    request_id: Option<String>,
    stream: GatewayByteStream,
}

impl GatewayTransport {
    /// Build a gateway transport from client-side gateway configuration.
    pub fn new(config: GatewayConfig) -> GatewayResult<Self> {
        let base_url = config.base_url.trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway base_url must not be empty",
            ));
        }
        validate_base_url(&base_url)?;
        if config.token.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::Authentication,
                "gateway bearer token must not be empty",
            ));
        }
        let timeout = config.timeout.unwrap_or(DEFAULT_TIMEOUT);
        if timeout.is_zero() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway timeout must be greater than zero",
            ));
        }

        let http = reqwest::Client::builder()
            .connect_timeout(timeout)
            .read_timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| {
                GatewayError::new(
                    GatewayErrorKind::Transport,
                    format!("failed to build HTTP client: {error}"),
                )
            })?;
        let headers = build_headers(config.token.expose())?;

        Ok(Self {
            http,
            base_url,
            headers,
            token: config.token,
            timeout,
        })
    }

    /// Run a non-streaming raw-prompt gateway request.
    pub async fn query(&self, req: GatewayQueryRequest) -> GatewayResult<GatewayTextResponse> {
        let body = query_body(&req, false)?;
        let response = self.post_json("/v1/query", &body).await?;
        text_response_from_body(response.request_id, response.body)
    }

    /// Run a streaming raw-prompt gateway request.
    pub async fn stream_query(
        &self,
        req: GatewayQueryRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        let body = query_body(&req, true)?;
        let response = self.post_json_stream("/v1/query", &body).await?;
        Ok(gateway_stream_events(
            response.request_id,
            response.stream,
            self.token.expose().to_string(),
        ))
    }

    /// Run a non-streaming chat gateway request.
    pub async fn chat(&self, req: GatewayChatRequest) -> GatewayResult<GatewayTextResponse> {
        let body = chat_body(&req, false)?;
        let response = self.post_json("/v1/chat", &body).await?;
        text_response_from_body(response.request_id, response.body)
    }

    /// Run a streaming chat gateway request.
    pub async fn stream_chat(
        &self,
        req: GatewayChatRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        let body = chat_body(&req, true)?;
        let response = self.post_json_stream("/v1/chat", &body).await?;
        Ok(gateway_stream_events(
            response.request_id,
            response.stream,
            self.token.expose().to_string(),
        ))
    }

    /// Run an embedding gateway request.
    pub async fn embed(&self, req: GatewayEmbedRequest) -> GatewayResult<GatewayEmbeddingResponse> {
        let body = embed_body(&req)?;
        let response = self.post_json("/v1/embed", &body).await?;
        embedding_response_from_body(response.request_id, response.body)
    }

    async fn post_json<T: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
    ) -> GatewayResult<HttpResponse> {
        self.send(self.http.post(self.url(path)).json(body)).await
    }

    async fn post_json_stream<T: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
    ) -> GatewayResult<HttpStreamResponse> {
        self.send_stream(self.http.post(self.url(path)).json(body))
            .await
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> GatewayResult<HttpResponse> {
        let response = request
            .headers(self.headers.clone())
            .timeout(self.timeout)
            .send()
            .await
            .map_err(transport_error)?;
        let status = response.status();
        let request_id = request_id(response.headers());
        let retry_after = retry_after(response.headers());

        if status.is_success() {
            let body = response
                .json::<serde_json::Value>()
                .await
                .map_err(transport_error)?;
            return Ok(HttpResponse { request_id, body });
        }

        let body = error_body(response).await;
        Err(gateway_error(
            status,
            request_id,
            retry_after,
            body,
            self.token.expose(),
        ))
    }

    async fn send_stream(
        &self,
        request: reqwest::RequestBuilder,
    ) -> GatewayResult<HttpStreamResponse> {
        let response = request
            .headers(self.headers.clone())
            .timeout(self.timeout)
            .send()
            .await
            .map_err(transport_error)?;
        let status = response.status();
        let request_id = request_id(response.headers());
        let retry_after = retry_after(response.headers());

        if !status.is_success() {
            let body = error_body(response).await;
            return Err(gateway_error(
                status,
                request_id,
                retry_after,
                body,
                self.token.expose(),
            ));
        }

        let stream = response
            .bytes_stream()
            .map(|item| item.map_err(transport_error));

        Ok(HttpStreamResponse {
            request_id,
            stream: Box::pin(stream),
        })
    }
}

pub(crate) fn build_headers(token: &str) -> GatewayResult<HeaderMap> {
    let mut headers = HeaderMap::new();
    let mut value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|error| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("invalid bearer token header: {error}"),
        )
    })?;
    value.set_sensitive(true);
    headers.insert(AUTHORIZATION, value);
    Ok(headers)
}

fn validate_base_url(base_url: &str) -> GatewayResult<()> {
    let url = reqwest::Url::parse(base_url).map_err(|error| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("gateway base_url is invalid: {error}"),
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "gateway base_url must be an absolute http(s) URL",
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "gateway base_url must not include userinfo",
        ));
    }
    if url.scheme() == "http" && !is_loopback_url(&url) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "gateway base_url must use HTTPS unless it targets loopback",
        ));
    }
    Ok(())
}

fn is_loopback_url(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host
        .trim_matches(|c| c == '[' || c == ']')
        .parse::<IpAddr>()
    {
        Ok(addr) => addr.is_loopback(),
        Err(_) => false,
    }
}

async fn error_body(response: reqwest::Response) -> serde_json::Value {
    match response.bytes().await {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned())
        }),
        Err(_) => serde_json::Value::Null,
    }
}

fn gateway_error(
    status: reqwest::StatusCode,
    request_id: Option<String>,
    retry_after: Option<Duration>,
    raw: serde_json::Value,
    redaction_secret: &str,
) -> GatewayError {
    let message = raw
        .pointer("/error/message")
        .and_then(serde_json::Value::as_str)
        .or_else(|| raw.get("message").and_then(serde_json::Value::as_str))
        .unwrap_or_else(|| status.canonical_reason().unwrap_or("gateway error"))
        .to_string();
    let code = raw
        .pointer("/error/code")
        .or_else(|| raw.pointer("/error/type"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let kind = gateway_error_kind(status, code.as_deref());

    let mut error = GatewayError {
        kind,
        status: Some(status.as_u16()),
        code,
        message,
        retry_after,
        request_id,
        raw: Some(Box::new(raw)),
    };
    error.redact_secret(redaction_secret);
    error
}

fn gateway_error_kind(status: reqwest::StatusCode, code: Option<&str>) -> GatewayErrorKind {
    if let Some(kind) = gateway_error_kind_from_code(code) {
        return kind;
    }

    match status.as_u16() {
        401 => GatewayErrorKind::Authentication,
        402 => GatewayErrorKind::QuotaExceeded,
        403 => GatewayErrorKind::Authorization,
        404 => GatewayErrorKind::ModelNotFound,
        408 => GatewayErrorKind::Timeout,
        429 => GatewayErrorKind::RateLimited,
        500 | 502 | 503 | 504 | 529 => GatewayErrorKind::Overloaded,
        400..=499 => GatewayErrorKind::InvalidRequest,
        _ => GatewayErrorKind::Gateway,
    }
}

fn transport_error(error: reqwest::Error) -> GatewayError {
    let kind = if error.is_timeout() {
        GatewayErrorKind::Timeout
    } else {
        GatewayErrorKind::Transport
    };
    GatewayError::new(kind, error.to_string())
}

fn request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-request-id")
        .or_else(|| headers.get("request-id"))
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get("retry-after-ms")
        .and_then(parse_duration_header_ms)
        .or_else(|| {
            headers
                .get("retry-after")
                .and_then(parse_duration_header_secs)
        })
}

fn parse_duration_header_ms(value: &HeaderValue) -> Option<Duration> {
    value
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_millis)
}

fn parse_duration_header_secs(value: &HeaderValue) -> Option<Duration> {
    value
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}
