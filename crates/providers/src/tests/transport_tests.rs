//! Tests the `transport` module in `cogentlm-providers`.
//!
//! Covers HTTP transport construction, header/auth validation, request metadata,
//! error classification, and stream response handling with deterministic
//! `wiremock` fixtures and no live network calls.

use super::*;
use futures_util::StreamExt;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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

#[test]
fn transport_rejects_empty_base_url_zero_timeout_and_non_http_urls() {
    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "",
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            Vec::new(),
            None,
        ),
        "empty base url should fail",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "ftp://example.com",
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            Vec::new(),
            None,
        ),
        "non-http url should fail",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "http://localhost",
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            Vec::new(),
            Some(Duration::ZERO),
        ),
        "zero timeout should fail",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn transport_rejects_invalid_auth_and_static_headers() {
    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "http://localhost",
            ProviderAuth::Bearer(crate::SecretString::new("")),
            Vec::new(),
            None,
        ),
        "empty bearer token",
    );
    assert_eq!(err.kind, ProviderErrorKind::Authentication);

    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "http://localhost",
            ProviderAuth::Bearer(crate::SecretString::new("bad\nvalue")),
            Vec::new(),
            None,
        ),
        "invalid bearer token",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "http://localhost",
            ProviderAuth::Header {
                name: "x-api-key".to_string(),
                value: crate::SecretString::new(""),
            },
            Vec::new(),
            None,
        ),
        "empty auth header value",
    );
    assert_eq!(err.kind, ProviderErrorKind::Authentication);

    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "http://localhost",
            ProviderAuth::Header {
                name: "bad header".to_string(),
                value: crate::SecretString::new("token"),
            },
            Vec::new(),
            None,
        ),
        "invalid auth header name",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "http://localhost",
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            vec![("bad header".to_string(), "value".to_string())],
            None,
        ),
        "invalid static header name",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = expect_provider_err(
        HttpTransport::new_with_options(
            ProviderKind::Proxy,
            "http://localhost",
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            vec![("x-test".to_string(), "bad\nvalue".to_string())],
            None,
        ),
        "invalid static header value",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[tokio::test]
async fn get_json_sends_auth_and_static_headers_and_reads_request_id() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer token"))
        .and(header("x-static", "yes"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-1")
                .set_body_json(json!({ "ok": true })),
        )
        .mount(&server)
        .await;

    let transport = HttpTransport::new_with_options(
        ProviderKind::Proxy,
        format!("{}/", server.uri()),
        ProviderAuth::Bearer(crate::SecretString::new("token")),
        vec![("x-static".to_string(), "yes".to_string())],
        None,
    )
    .expect("transport");
    let response = transport.get_json("/models").await.expect("response");

    assert_eq!(response.request_id.as_deref(), Some("req-1"));
    assert_eq!(response.body["ok"], true);
}

#[tokio::test]
async fn post_json_reads_request_id_fallback_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat"))
        .and(header("authorization", "Bearer token"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("request-id", "req-fallback")
                .set_body_json(json!({ "ok": true })),
        )
        .mount(&server)
        .await;

    let response = transport(&server)
        .post_json("/chat", &json!({ "prompt": "hello" }))
        .await
        .expect("response");

    assert_eq!(response.request_id.as_deref(), Some("req-fallback"));
}

#[tokio::test]
async fn success_with_invalid_json_is_transport_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bad-json"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let err = expect_provider_err(
        transport(&server).get_json("/bad-json").await,
        "invalid success body should fail",
    );

    assert_eq!(err.kind, ProviderErrorKind::Transport);
}

#[tokio::test]
async fn provider_http_errors_preserve_metadata_and_body_message_shapes() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(404)
                .insert_header("request-id", "req-missing")
                .insert_header("retry-after", "3")
                .set_body_json(json!({
                    "error": {
                        "message": "missing model",
                        "type": "not_found_error"
                    }
                })),
        )
        .mount(&server)
        .await;

    let err = expect_provider_err(
        transport(&server).get_json("/models").await,
        "404 should fail",
    );

    assert_eq!(err.kind, ProviderErrorKind::ModelNotFound);
    assert_eq!(err.status, Some(404));
    assert_eq!(err.code.as_deref(), Some("not_found_error"));
    assert_eq!(err.message, "missing model");
    assert_eq!(err.request_id.as_deref(), Some("req-missing"));
    assert_eq!(err.retry_after, Some(Duration::from_secs(3)));
    assert!(err.raw.is_some());
}

#[tokio::test]
async fn provider_http_errors_use_top_level_message_or_status_reason() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/top-level"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "message": "top-level bad request"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/status-reason"))
        .respond_with(ResponseTemplate::new(418).set_body_json(json!({})))
        .mount(&server)
        .await;

    let err = expect_provider_err(
        transport(&server).get_json("/top-level").await,
        "400 should fail",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert_eq!(err.message, "top-level bad request");

    let err = expect_provider_err(
        transport(&server).get_json("/status-reason").await,
        "418 should fail",
    );
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert!(err.message.contains("teapot"));
}

#[tokio::test]
async fn post_json_stream_returns_request_id_and_body_bytes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-stream")
                .set_body_string("data: hi\n\n"),
        )
        .mount(&server)
        .await;

    let mut response = transport(&server)
        .post_json_stream("/stream", &json!({ "stream": true }))
        .await
        .expect("stream response");

    assert_eq!(response.request_id.as_deref(), Some("req-stream"));
    let bytes = response
        .stream
        .next()
        .await
        .expect("stream chunk")
        .expect("chunk bytes");
    assert_eq!(&bytes[..], b"data: hi\n\n");
}

#[tokio::test]
async fn post_json_stream_maps_non_success_response_before_streaming() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/stream"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("retry-after", "5")
                .set_body_string("busy"),
        )
        .mount(&server)
        .await;

    let err = expect_provider_err(
        transport(&server)
            .post_json_stream("/stream", &json!({ "stream": true }))
            .await,
        "503 stream response should fail",
    );

    assert_eq!(err.kind, ProviderErrorKind::Overloaded);
    assert_eq!(err.status, Some(503));
    assert_eq!(err.retry_after, Some(Duration::from_secs(5)));
    assert_eq!(
        err.raw.as_deref(),
        Some(&serde_json::Value::String("busy".to_string()))
    );
}

#[test]
fn provider_error_kind_maps_status_fallbacks() {
    let cases = [
        (
            reqwest::StatusCode::UNAUTHORIZED,
            ProviderErrorKind::Authentication,
        ),
        (
            reqwest::StatusCode::PAYMENT_REQUIRED,
            ProviderErrorKind::QuotaExceeded,
        ),
        (
            reqwest::StatusCode::FORBIDDEN,
            ProviderErrorKind::Authorization,
        ),
        (
            reqwest::StatusCode::NOT_FOUND,
            ProviderErrorKind::ModelNotFound,
        ),
        (
            reqwest::StatusCode::REQUEST_TIMEOUT,
            ProviderErrorKind::Timeout,
        ),
        (
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            ProviderErrorKind::RateLimited,
        ),
        (
            reqwest::StatusCode::BAD_GATEWAY,
            ProviderErrorKind::Overloaded,
        ),
        (
            reqwest::StatusCode::BAD_REQUEST,
            ProviderErrorKind::InvalidRequest,
        ),
        (
            reqwest::StatusCode::IM_A_TEAPOT,
            ProviderErrorKind::InvalidRequest,
        ),
        (reqwest::StatusCode::CONTINUE, ProviderErrorKind::Provider),
    ];

    for (status, expected) in cases {
        assert_eq!(provider_error_kind(status, None), expected);
    }
}

#[test]
fn retry_after_and_request_id_ignore_invalid_header_values() {
    let mut headers = HeaderMap::new();
    headers.insert("retry-after-ms", HeaderValue::from_static("not-a-number"));
    headers.insert("retry-after", HeaderValue::from_static("also-bad"));
    headers.insert(
        "x-request-id",
        HeaderValue::from_bytes(b"\xff").expect("non-utf8 header value"),
    );

    assert_eq!(retry_after(&headers), None);
    assert_eq!(request_id(&headers), None);
}

fn transport(server: &MockServer) -> HttpTransport {
    HttpTransport::new_with_options(
        ProviderKind::Proxy,
        server.uri(),
        ProviderAuth::Bearer(crate::SecretString::new("token")),
        Vec::new(),
        None,
    )
    .expect("transport")
}

fn expect_provider_err<T>(result: ProviderResult<T>, message: &str) -> ProviderError {
    match result {
        Ok(_) => panic!("{message}"),
        Err(err) => err,
    }
}
