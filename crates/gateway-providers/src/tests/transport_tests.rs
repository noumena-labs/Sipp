//! Unit tests for the parent module.

use super::*;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

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
        ProviderKind::OpenAiCompatible,
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
fn transport_rejects_plain_http_non_loopback_base_url() {
    let err = match HttpTransport::new_with_options(
        ProviderKind::OpenAiCompatible,
        "http://example.com",
        ProviderAuth::Bearer(crate::SecretString::new("token")),
        Vec::new(),
        None,
    ) {
        Ok(_) => panic!("plain HTTP upstream should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert_eq!(
        err.message,
        "provider base_url must use HTTPS unless it targets loopback"
    );
}

#[test]
fn transport_rejects_base_url_surrounding_whitespace() {
    for base_url in [
        " https://example.com/v1",
        "https://example.com/v1 ",
        "https://example.com/v1/ ",
    ] {
        let err = match HttpTransport::new_with_options(
            ProviderKind::OpenAiCompatible,
            base_url,
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            Vec::new(),
            None,
        ) {
            Ok(_) => panic!("base URL whitespace should fail"),
            Err(err) => err,
        };

        assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
        assert_eq!(
            err.message,
            "provider base_url must not contain surrounding whitespace"
        );
    }
}

#[test]
fn transport_rejects_base_url_userinfo() {
    let err = match HttpTransport::new_with_options(
        ProviderKind::OpenAiCompatible,
        "https://user:provider-secret@example.com/v1",
        ProviderAuth::Bearer(crate::SecretString::new("token")),
        Vec::new(),
        None,
    ) {
        Ok(_) => panic!("base URL userinfo should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert_eq!(err.message, "provider base_url must not include userinfo");
    assert!(!format!("{err:?}").contains("provider-secret"));
}

#[test]
fn transport_rejects_base_url_query_and_fragment() {
    for base_url in [
        "https://example.com/v1?token=provider-secret",
        "https://example.com/v1#provider-secret",
    ] {
        let err = match HttpTransport::new_with_options(
            ProviderKind::OpenAiCompatible,
            base_url,
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            Vec::new(),
            None,
        ) {
            Ok(_) => panic!("base URL query and fragment should fail"),
            Err(err) => err,
        };

        assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
        assert_eq!(
            err.message,
            "provider base_url must not include query or fragment"
        );
        assert!(!format!("{err:?}").contains("provider-secret"));
    }
}

#[test]
fn transport_allows_plain_http_loopback_base_url() {
    for base_url in ["http://localhost:8080", "http://127.0.0.1:8080"] {
        HttpTransport::new_with_options(
            ProviderKind::OpenAiCompatible,
            base_url,
            ProviderAuth::Bearer(crate::SecretString::new("token")),
            Vec::new(),
            None,
        )
        .expect("loopback HTTP upstream should be allowed for development");
    }
}

#[test]
fn static_provider_headers_are_sensitive() {
    let headers = parse_static_headers(
        ProviderKind::OpenAiCompatible,
        vec![("x-provider-secret".to_string(), "secret-value".to_string())],
    )
    .expect("headers");

    assert!(headers
        .get("x-provider-secret")
        .expect("static header")
        .is_sensitive());
}

#[test]
fn provider_auth_rejects_blank_or_whitespace_secrets() {
    let blank = build_request_headers(
        ProviderKind::OpenAiCompatible,
        ProviderAuth::Bearer(crate::SecretString::new(" \t ")),
        Vec::new(),
    )
    .expect_err("blank bearer token must fail");
    assert_eq!(blank.kind, ProviderErrorKind::Authentication);
    assert_eq!(blank.message, "bearer token must not be empty");

    let whitespace = build_request_headers(
        ProviderKind::OpenAiCompatible,
        ProviderAuth::Bearer(crate::SecretString::new("secret token")),
        Vec::new(),
    )
    .expect_err("whitespace bearer token must fail");
    assert_eq!(whitespace.kind, ProviderErrorKind::InvalidRequest);
    assert_eq!(
        whitespace.message,
        "bearer token must not contain whitespace"
    );
    assert!(!format!("{whitespace:?}").contains("secret token"));

    let header_value = build_request_headers(
        ProviderKind::OpenAiCompatible,
        ProviderAuth::Header {
            name: "x-api-key".to_string(),
            value: crate::SecretString::new("secret token"),
        },
        Vec::new(),
    )
    .expect_err("whitespace header value must fail");
    assert_eq!(header_value.kind, ProviderErrorKind::InvalidRequest);
    assert_eq!(
        header_value.message,
        "auth header value must not contain whitespace"
    );
    assert!(!format!("{header_value:?}").contains("secret token"));
}

#[tokio::test]
async fn transport_does_not_follow_provider_redirects() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(307)
                .insert_header("location", format!("{}/redirected", server.uri())),
        )
        .mount(&server)
        .await;

    let transport = HttpTransport::new_with_options(
        ProviderKind::OpenAiCompatible,
        server.uri(),
        ProviderAuth::Bearer(crate::SecretString::new("token")),
        Vec::new(),
        None,
    )
    .expect("transport");

    let err = match transport.get_json("/models").await {
        Ok(_) => panic!("redirect should not be followed"),
        Err(err) => err,
    };

    assert_eq!(err.status, Some(307));
    assert_eq!(err.kind, ProviderErrorKind::Provider);
}

#[tokio::test]
async fn provider_http_error_body_is_capped() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(500).set_body_string("x".repeat((1 << 20) + 1)))
        .mount(&server)
        .await;

    let transport = HttpTransport::new_with_options(
        ProviderKind::OpenAiCompatible,
        server.uri(),
        ProviderAuth::Bearer(crate::SecretString::new("token")),
        Vec::new(),
        None,
    )
    .expect("transport");

    let err = match transport.get_json("/models").await {
        Ok(_) => panic!("huge provider error should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind, ProviderErrorKind::Overloaded);
    assert_eq!(err.status, Some(500));
    assert_eq!(err.message, "provider error response exceeded body limit");
    assert!(!format!("{err:?}").contains(&"x".repeat(1024)));
}
