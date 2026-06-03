//! Tests the `error` module in `cogentlm-providers`.
//!
//! Covers provider error kind wire labels, provider error construction, and
//! provider error-code normalization with deterministic values and no HTTP calls.

use super::*;

#[test]
fn provider_error_kind_from_code_maps_retry_relevant_codes() {
    assert_eq!(
        provider_error_kind_from_code(Some("rate_limit")),
        Some(ProviderErrorKind::RateLimited)
    );
    assert_eq!(
        provider_error_kind_from_code(Some("insufficient_quota")),
        Some(ProviderErrorKind::QuotaExceeded)
    );
    assert_eq!(
        provider_error_kind_from_code(Some("overloaded_error")),
        Some(ProviderErrorKind::Overloaded)
    );
    assert_eq!(provider_error_kind_from_code(Some("bad_request")), None);
}

#[test]
fn provider_error_kind_labels_cover_all_public_variants() {
    assert_eq!(ProviderErrorKind::Authentication.as_str(), "authentication");
    assert_eq!(ProviderErrorKind::Authorization.as_str(), "authorization");
    assert_eq!(ProviderErrorKind::RateLimited.as_str(), "rate_limited");
    assert_eq!(ProviderErrorKind::QuotaExceeded.as_str(), "quota_exceeded");
    assert_eq!(
        ProviderErrorKind::InvalidRequest.as_str(),
        "invalid_request"
    );
    assert_eq!(
        ProviderErrorKind::UnsupportedFeature.as_str(),
        "unsupported_feature"
    );
    assert_eq!(ProviderErrorKind::ModelNotFound.as_str(), "model_not_found");
    assert_eq!(ProviderErrorKind::Timeout.as_str(), "timeout");
    assert_eq!(ProviderErrorKind::Overloaded.as_str(), "overloaded");
    assert_eq!(ProviderErrorKind::Transport.as_str(), "transport");
    assert_eq!(ProviderErrorKind::Provider.as_str(), "provider");
}

#[test]
fn provider_error_new_sets_only_core_fields() {
    let err = ProviderError::new(
        ProviderErrorKind::InvalidRequest,
        ProviderKind::OpenAi,
        "bad request",
    );

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert_eq!(err.provider, ProviderKind::OpenAi);
    assert_eq!(err.status, None);
    assert_eq!(err.code, None);
    assert_eq!(err.message, "bad request");
    assert_eq!(err.retry_after, None);
    assert_eq!(err.request_id, None);
    assert!(err.raw.is_none());
    assert!(err.to_string().contains("bad request"));
}
