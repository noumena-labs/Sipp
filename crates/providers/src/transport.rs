use std::{net::IpAddr, time::Duration};

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION};

use crate::{
    error::provider_error_kind_from_code, ProviderAuth, ProviderError, ProviderErrorKind,
    ProviderKind, ProviderRequestContext, ProviderResult,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "tests/transport_tests.rs"]
mod transport_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_ERROR_BODY_BYTES: usize = 1 << 20;

#[derive(Clone)]
pub(crate) struct HttpTransport {
    http: reqwest::Client,
    base_url: String,
    provider: ProviderKind,
    headers: HeaderMap,
    timeout: Duration,
}

pub(crate) struct HttpResponse {
    pub(crate) request_id: Option<String>,
    pub(crate) body: serde_json::Value,
}

pub(crate) type HttpByteStream =
    std::pin::Pin<Box<dyn Stream<Item = ProviderResult<Bytes>> + Send>>;

pub(crate) struct HttpStreamResponse {
    pub(crate) request_id: Option<String>,
    pub(crate) stream: HttpByteStream,
}

impl HttpTransport {
    pub(crate) fn new_with_options(
        provider: ProviderKind,
        base_url: impl Into<String>,
        auth: ProviderAuth,
        static_headers: Vec<(String, String)>,
        timeout: Option<Duration>,
    ) -> ProviderResult<Self> {
        let base_url = base_url.into();
        let trimmed_base_url = base_url.trim();
        if trimmed_base_url.is_empty() {
            return Err(ProviderError::new(
                ProviderErrorKind::InvalidRequest,
                provider,
                "provider base_url must not be empty",
            ));
        }
        if trimmed_base_url != base_url.as_str() {
            return Err(ProviderError::new(
                ProviderErrorKind::InvalidRequest,
                provider,
                "provider base_url must not contain surrounding whitespace",
            ));
        }
        let base_url = base_url.trim_end_matches('/').to_string();
        validate_base_url(provider, &base_url)?;
        let timeout = timeout.unwrap_or(DEFAULT_TIMEOUT);
        if timeout.is_zero() {
            return Err(ProviderError::new(
                ProviderErrorKind::InvalidRequest,
                provider,
                "provider timeout must be greater than zero",
            ));
        }

        // No *total* request timeout on the HTTP handle: a total deadline is the wrong
        // shape for incremental responses and would abort a long generation. A
        // connect timeout bounds connection setup and a read timeout bounds idle
        // gaps between chunks; unary calls additionally get a per-request total
        // timeout in `send`.
        let http = reqwest::Client::builder()
            .connect_timeout(timeout)
            .read_timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| {
                ProviderError::new(
                    ProviderErrorKind::Transport,
                    provider,
                    format!("failed to build HTTP client: {err}"),
                )
            })?;
        let headers = build_request_headers(provider, auth, static_headers)?;

        Ok(Self {
            http,
            base_url,
            provider,
            headers,
            timeout,
        })
    }

    pub(crate) async fn get_json(&self, path: &str) -> ProviderResult<HttpResponse> {
        self.send(self.http.get(self.url(path))).await
    }

    pub(crate) async fn post_json<T: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
    ) -> ProviderResult<HttpResponse> {
        self.send(self.http.post(self.url(path)).json(body)).await
    }

    pub(crate) async fn post_json_with_context<T: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
        context: &ProviderRequestContext,
        correlation_header: Option<&str>,
    ) -> ProviderResult<HttpResponse> {
        let request = self.http.post(self.url(path)).json(body);
        self.send(with_correlation(
            request,
            self.provider,
            context,
            correlation_header,
        )?)
        .await
    }

    pub(crate) async fn post_json_stream<T: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
    ) -> ProviderResult<HttpStreamResponse> {
        self.send_stream(self.http.post(self.url(path)).json(body))
            .await
    }

    pub(crate) async fn post_json_stream_with_context<T: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
        context: &ProviderRequestContext,
        correlation_header: Option<&str>,
    ) -> ProviderResult<HttpStreamResponse> {
        let request = self.http.post(self.url(path)).json(body);
        self.send_stream(with_correlation(
            request,
            self.provider,
            context,
            correlation_header,
        )?)
        .await
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> ProviderResult<HttpResponse> {
        let response = request
            .headers(self.headers.clone())
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|err| self.transport_error(err))?;
        let status = response.status();
        let request_id = request_id(response.headers());
        let retry_after = retry_after(response.headers());

        if status.is_success() {
            let body = response
                .json::<serde_json::Value>()
                .await
                .map_err(|err| self.transport_error(err))?;
            return Ok(HttpResponse { request_id, body });
        }

        let body = error_body(response).await;
        Err(self.provider_error(status, request_id, retry_after, body))
    }

    async fn send_stream(
        &self,
        request: reqwest::RequestBuilder,
    ) -> ProviderResult<HttpStreamResponse> {
        // Incremental responses use the HTTP idle read timeout, not a total deadline, so
        // a long-but-progressing generation is not aborted mid-stream.
        let response = request
            .headers(self.headers.clone())
            .send()
            .await
            .map_err(|err| self.transport_error(err))?;
        let status = response.status();
        let request_id = request_id(response.headers());
        let retry_after = retry_after(response.headers());

        if !status.is_success() {
            let body = error_body(response).await;
            return Err(self.provider_error(status, request_id, retry_after, body));
        }

        let provider = self.provider;
        let stream = response
            .bytes_stream()
            .map(move |item| item.map_err(|err| transport_error_for(provider, err)));

        Ok(HttpStreamResponse {
            request_id,
            stream: Box::pin(stream),
        })
    }

    fn transport_error(&self, err: reqwest::Error) -> ProviderError {
        transport_error_for(self.provider, err)
    }

    fn provider_error(
        &self,
        status: reqwest::StatusCode,
        request_id: Option<String>,
        retry_after: Option<Duration>,
        raw: serde_json::Value,
    ) -> ProviderError {
        let message = raw
            .pointer("/error/message")
            .and_then(serde_json::Value::as_str)
            .or_else(|| raw.get("message").and_then(serde_json::Value::as_str))
            .unwrap_or_else(|| status.canonical_reason().unwrap_or("provider error"))
            .to_string();
        let code = raw
            .pointer("/error/code")
            .or_else(|| raw.pointer("/error/type"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let kind = provider_error_kind(status, code.as_deref());

        ProviderError {
            kind,
            provider: self.provider,
            status: Some(status.as_u16()),
            code,
            message,
            retry_after,
            request_id,
            raw: Some(Box::new(raw)),
        }
    }
}

fn with_correlation(
    request: reqwest::RequestBuilder,
    provider: ProviderKind,
    context: &ProviderRequestContext,
    header: Option<&str>,
) -> ProviderResult<reqwest::RequestBuilder> {
    let (Some(request_id), Some(header)) = (context.request_id.as_deref(), header) else {
        return Ok(request);
    };
    let name = HeaderName::from_bytes(header.as_bytes()).map_err(|error| {
        ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            format!("invalid provider correlation header: {error}"),
        )
    })?;
    let value = HeaderValue::from_str(request_id).map_err(|error| {
        ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            format!("invalid provider request ID: {error}"),
        )
    })?;
    Ok(request.header(name, value))
}

/// Build the full request header map once at construction: static headers plus
/// the auth header, marked sensitive so it is not logged by the HTTP stack.
/// Auth validity is checked here so per-request sends cannot fail on auth.
fn build_request_headers(
    provider: ProviderKind,
    auth: ProviderAuth,
    static_headers: Vec<(String, String)>,
) -> ProviderResult<HeaderMap> {
    let mut headers = parse_static_headers(provider, static_headers)?;
    let (name, mut value) = match auth {
        ProviderAuth::Bearer(secret) => {
            if secret.is_blank() {
                return Err(ProviderError::new(
                    ProviderErrorKind::Authentication,
                    provider,
                    "bearer token must not be empty",
                ));
            }
            if secret.contains_whitespace() {
                return Err(ProviderError::new(
                    ProviderErrorKind::InvalidRequest,
                    provider,
                    "bearer token must not contain whitespace",
                ));
            }
            let value = HeaderValue::from_str(&format!("Bearer {}", secret.expose()))
                .map_err(|err| invalid_auth_header_error(provider, err))?;
            (AUTHORIZATION, value)
        }
        ProviderAuth::Header { name, value } => {
            if value.is_blank() {
                return Err(ProviderError::new(
                    ProviderErrorKind::Authentication,
                    provider,
                    "auth header value must not be empty",
                ));
            }
            if value.contains_whitespace() {
                return Err(ProviderError::new(
                    ProviderErrorKind::InvalidRequest,
                    provider,
                    "auth header value must not contain whitespace",
                ));
            }
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|err| invalid_auth_header_error(provider, err))?;
            let value = HeaderValue::from_str(value.expose())
                .map_err(|err| invalid_auth_header_error(provider, err))?;
            (name, value)
        }
    };
    value.set_sensitive(true);
    headers.insert(name, value);
    Ok(headers)
}

fn invalid_auth_header_error(provider: ProviderKind, err: impl std::fmt::Display) -> ProviderError {
    ProviderError::new(
        ProviderErrorKind::InvalidRequest,
        provider,
        format!("invalid auth header: {err}"),
    )
}

/// Read an error response body without requiring valid JSON. Gateways, proxies,
/// and CDNs frequently return HTML or plain text on 4xx/5xx; the status-based
/// classification stays intact and the raw body is preserved either way.
async fn error_body(response: reqwest::Response) -> serde_json::Value {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else {
            return serde_json::Value::Null;
        };
        if body.len().saturating_add(chunk.len()) > MAX_ERROR_BODY_BYTES {
            return serde_json::json!({
                "error": {
                    "message": "provider error response exceeded body limit"
                }
            });
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body)
        .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&body).into_owned()))
}

fn parse_static_headers(
    provider: ProviderKind,
    headers: Vec<(String, String)>,
) -> ProviderResult<HeaderMap> {
    let mut output = HeaderMap::new();
    for (name, value) in headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|err| invalid_header_error_for(provider, err))?;
        let mut value =
            HeaderValue::from_str(&value).map_err(|err| invalid_header_error_for(provider, err))?;
        value.set_sensitive(true);
        output.insert(name, value);
    }
    Ok(output)
}

fn validate_base_url(provider: ProviderKind, base_url: &str) -> ProviderResult<()> {
    let url = reqwest::Url::parse(base_url).map_err(|err| {
        ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            format!("provider base_url is invalid: {err}"),
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            "provider base_url must be an absolute http(s) URL",
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            "provider base_url must not include userinfo",
        ));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            "provider base_url must not include query or fragment",
        ));
    }
    if url.scheme() == "http" && !is_loopback_url(&url) {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            "provider base_url must use HTTPS unless it targets loopback",
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

fn request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-request-id")
        .or_else(|| headers.get("request-id"))
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn provider_error_kind(status: reqwest::StatusCode, code: Option<&str>) -> ProviderErrorKind {
    if let Some(kind) = provider_error_kind_from_code(code) {
        return kind;
    }

    match status.as_u16() {
        401 => ProviderErrorKind::Authentication,
        402 => ProviderErrorKind::QuotaExceeded,
        403 => ProviderErrorKind::Authorization,
        404 => ProviderErrorKind::ModelNotFound,
        408 => ProviderErrorKind::Timeout,
        429 => ProviderErrorKind::RateLimited,
        500 | 502 | 503 | 504 | 529 => ProviderErrorKind::Overloaded,
        400..=499 => ProviderErrorKind::InvalidRequest,
        _ => ProviderErrorKind::Provider,
    }
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

fn invalid_header_error_for(provider: ProviderKind, err: impl std::fmt::Display) -> ProviderError {
    ProviderError::new(
        ProviderErrorKind::InvalidRequest,
        provider,
        format!("invalid provider header: {err}"),
    )
}

fn transport_error_for(provider: ProviderKind, err: reqwest::Error) -> ProviderError {
    let kind = if err.is_timeout() {
        ProviderErrorKind::Timeout
    } else {
        ProviderErrorKind::Transport
    };
    ProviderError::new(kind, provider, err.to_string())
}
