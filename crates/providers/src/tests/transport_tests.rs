//! Unit tests for the parent module.

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
