use std::time::Duration;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION};

use crate::{
    error::provider_error_kind_from_code, ProviderAuth, ProviderError, ProviderErrorKind,
    ProviderKind, ProviderResult,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub(crate) struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    provider: ProviderKind,
    auth: ProviderAuth,
    static_headers: HeaderMap,
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
        let base_url = base_url.into().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(ProviderError::new(
                ProviderErrorKind::InvalidRequest,
                provider,
                "provider base_url must not be empty",
            ));
        }
        validate_base_url(provider, &base_url)?;
        let timeout = timeout.unwrap_or(DEFAULT_TIMEOUT);
        if timeout.is_zero() {
            return Err(ProviderError::new(
                ProviderErrorKind::InvalidRequest,
                provider,
                "provider timeout must be greater than zero",
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| {
                ProviderError::new(
                    ProviderErrorKind::Transport,
                    provider,
                    format!("failed to build HTTP client: {err}"),
                )
            })?;
        let static_headers = parse_static_headers(provider, static_headers)?;

        Ok(Self {
            client,
            base_url,
            provider,
            auth,
            static_headers,
        })
    }

    pub(crate) async fn get_json(&self, path: &str) -> ProviderResult<HttpResponse> {
        self.send(self.client.get(self.url(path))).await
    }

    pub(crate) async fn post_json<T: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
    ) -> ProviderResult<HttpResponse> {
        self.send(self.client.post(self.url(path)).json(body)).await
    }

    pub(crate) async fn post_json_stream<T: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &T,
    ) -> ProviderResult<HttpStreamResponse> {
        self.send_stream(self.client.post(self.url(path)).json(body))
            .await
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> ProviderResult<HttpResponse> {
        let response = request
            .headers(self.auth_headers()?)
            .send()
            .await
            .map_err(|err| self.transport_error(err))?;
        let status = response.status();
        let request_id = request_id(response.headers());
        let retry_after = retry_after(response.headers());
        let body = response
            .json::<serde_json::Value>()
            .await
            .map_err(|err| self.transport_error(err))?;

        if status.is_success() {
            return Ok(HttpResponse { request_id, body });
        }

        Err(self.provider_error(status, request_id, retry_after, body))
    }

    async fn send_stream(
        &self,
        request: reqwest::RequestBuilder,
    ) -> ProviderResult<HttpStreamResponse> {
        let response = request
            .headers(self.auth_headers()?)
            .send()
            .await
            .map_err(|err| self.transport_error(err))?;
        let status = response.status();
        let request_id = request_id(response.headers());
        let retry_after = retry_after(response.headers());

        if !status.is_success() {
            let body = response
                .json::<serde_json::Value>()
                .await
                .map_err(|err| self.transport_error(err))?;
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

    fn auth_headers(&self) -> ProviderResult<HeaderMap> {
        let mut headers = self.static_headers.clone();
        match &self.auth {
            ProviderAuth::Bearer(secret) => {
                if secret.is_empty() {
                    return Err(ProviderError::new(
                        ProviderErrorKind::Authentication,
                        self.provider,
                        "bearer token must not be empty",
                    ));
                }
                let value = HeaderValue::from_str(&format!("Bearer {}", secret.expose()))
                    .map_err(|err| self.invalid_header_error(err))?;
                headers.insert(AUTHORIZATION, value);
            }
            ProviderAuth::Header { name, value } => {
                let name = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|err| self.invalid_header_error(err))?;
                let value = HeaderValue::from_str(value.expose())
                    .map_err(|err| self.invalid_header_error(err))?;
                headers.insert(name, value);
            }
        }
        Ok(headers)
    }

    fn invalid_header_error(&self, err: impl std::fmt::Display) -> ProviderError {
        ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            self.provider,
            format!("invalid auth header: {err}"),
        )
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

fn parse_static_headers(
    provider: ProviderKind,
    headers: Vec<(String, String)>,
) -> ProviderResult<HeaderMap> {
    let mut output = HeaderMap::new();
    for (name, value) in headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|err| invalid_header_error_for(provider, err))?;
        let value =
            HeaderValue::from_str(&value).map_err(|err| invalid_header_error_for(provider, err))?;
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
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_prefers_milliseconds_then_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("2"));
        assert_eq!(retry_after(&headers), Some(Duration::from_secs(2)));

        headers.insert("retry-after-ms", HeaderValue::from_static("1500"));
        assert_eq!(retry_after(&headers), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn provider_error_kind_distinguishes_quota_from_rate_limit() {
        assert_eq!(
            provider_error_kind(reqwest::StatusCode::TOO_MANY_REQUESTS, Some("rate_limit")),
            ProviderErrorKind::RateLimited
        );
        assert_eq!(
            provider_error_kind(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                Some("insufficient_quota")
            ),
            ProviderErrorKind::QuotaExceeded
        );
        assert_eq!(
            provider_error_kind(reqwest::StatusCode::PAYMENT_REQUIRED, None),
            ProviderErrorKind::QuotaExceeded
        );
    }

    #[test]
    fn transport_rejects_invalid_base_url() {
        let err = match HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "not-a-url",
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            Vec::new(),
            None,
        ) {
            Ok(_) => panic!("invalid base url should fail"),
            Err(err) => err,
        };

        assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    }
}
