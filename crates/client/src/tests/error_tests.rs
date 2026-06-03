//! Tests the `error` module in `cogentlm-client`.
//!
//! Covers provider-to-client remote error classification and stable diagnostic
//! labels using synthetic provider errors without remote transport calls.

#[cfg(feature = "providers")]
use std::time::Duration;

#[cfg(feature = "providers")]
use cogentlm_providers::{ProviderError, ProviderErrorKind, ProviderKind};
#[cfg(feature = "providers")]
use serde_json::json;

#[cfg(feature = "providers")]
use super::*;

#[cfg(feature = "providers")]
#[test]
fn remote_kind_labels_are_stable() {
    assert_eq!(RemoteKind::Proxy.as_str(), "proxy");
    assert_eq!(RemoteKind::OpenAi.as_str(), "openai");
    assert_eq!(RemoteKind::Anthropic.as_str(), "anthropic");
}

#[cfg(feature = "providers")]
#[test]
fn remote_error_kind_labels_are_stable() {
    let cases = [
        (RemoteErrorKind::Authentication, "authentication"),
        (RemoteErrorKind::Authorization, "authorization"),
        (RemoteErrorKind::RateLimited, "rate_limited"),
        (RemoteErrorKind::QuotaExceeded, "quota_exceeded"),
        (RemoteErrorKind::InvalidRequest, "invalid_request"),
        (RemoteErrorKind::UnsupportedFeature, "unsupported_feature"),
        (RemoteErrorKind::ModelNotFound, "model_not_found"),
        (RemoteErrorKind::Timeout, "timeout"),
        (RemoteErrorKind::Overloaded, "overloaded"),
        (RemoteErrorKind::Transport, "transport"),
        (RemoteErrorKind::Remote, "remote"),
    ];

    for (kind, label) in cases {
        assert_eq!(kind.as_str(), label);
    }
}

#[cfg(feature = "providers")]
#[test]
fn remote_error_new_sets_required_fields_and_display() {
    let error = RemoteError::new(RemoteErrorKind::Timeout, RemoteKind::OpenAi, "slow");

    assert_eq!(error.kind, RemoteErrorKind::Timeout);
    assert_eq!(error.remote_kind, RemoteKind::OpenAi);
    assert_eq!(error.message, "slow");
    assert!(error.status.is_none());
    assert!(error.code.is_none());
    assert!(error.retry_after.is_none());
    assert!(error.request_id.is_none());
    assert!(error.raw.is_none());
    assert_eq!(error.to_string(), "openai remote error (timeout): slow");
}

#[cfg(feature = "providers")]
#[test]
fn provider_error_kind_conversion_covers_all_variants() {
    let cases = [
        (
            ProviderErrorKind::Authentication,
            RemoteErrorKind::Authentication,
        ),
        (
            ProviderErrorKind::Authorization,
            RemoteErrorKind::Authorization,
        ),
        (ProviderErrorKind::RateLimited, RemoteErrorKind::RateLimited),
        (
            ProviderErrorKind::QuotaExceeded,
            RemoteErrorKind::QuotaExceeded,
        ),
        (
            ProviderErrorKind::InvalidRequest,
            RemoteErrorKind::InvalidRequest,
        ),
        (
            ProviderErrorKind::UnsupportedFeature,
            RemoteErrorKind::UnsupportedFeature,
        ),
        (
            ProviderErrorKind::ModelNotFound,
            RemoteErrorKind::ModelNotFound,
        ),
        (ProviderErrorKind::Timeout, RemoteErrorKind::Timeout),
        (ProviderErrorKind::Overloaded, RemoteErrorKind::Overloaded),
        (ProviderErrorKind::Transport, RemoteErrorKind::Transport),
        (ProviderErrorKind::Provider, RemoteErrorKind::Remote),
    ];

    for (provider_kind, remote_kind) in cases {
        let error = ProviderError::new(provider_kind, ProviderKind::Proxy, "provider error");
        let remote = RemoteError::from(error);

        assert_eq!(remote.kind, remote_kind);
        assert_eq!(remote.remote_kind, RemoteKind::Proxy);
        assert_eq!(remote.message, "provider error");
    }
}

#[cfg(feature = "providers")]
#[test]
fn provider_metadata_is_preserved_when_converted() {
    let mut error = ProviderError::new(
        ProviderErrorKind::RateLimited,
        ProviderKind::Anthropic,
        "limited",
    );
    error.status = Some(429);
    error.code = Some("rate_limit_error".to_string());
    error.retry_after = Some(Duration::from_secs(2));
    error.request_id = Some("req-1".to_string());
    error.raw = Some(Box::new(json!({ "error": "limited" })));

    let remote = RemoteError::from(error);

    assert_eq!(remote.kind, RemoteErrorKind::RateLimited);
    assert_eq!(remote.remote_kind, RemoteKind::Anthropic);
    assert_eq!(remote.status, Some(429));
    assert_eq!(remote.code.as_deref(), Some("rate_limit_error"));
    assert_eq!(remote.retry_after, Some(Duration::from_secs(2)));
    assert_eq!(remote.request_id.as_deref(), Some("req-1"));
    assert_eq!(remote.raw.as_deref(), Some(&json!({ "error": "limited" })));
}

#[cfg(feature = "providers")]
#[test]
fn provider_kind_conversion_covers_all_remote_families() {
    let cases = [
        (ProviderKind::Proxy, RemoteKind::Proxy),
        (ProviderKind::OpenAi, RemoteKind::OpenAi),
        (ProviderKind::Anthropic, RemoteKind::Anthropic),
    ];

    for (provider, remote_kind) in cases {
        let error = ProviderError::new(ProviderErrorKind::Provider, provider, "remote");
        let remote = RemoteError::from(error);

        assert_eq!(remote.remote_kind, remote_kind);
    }
}

#[cfg(feature = "providers")]
#[test]
fn provider_error_converts_into_cogent_remote_error() {
    let error = ProviderError::new(ProviderErrorKind::Transport, ProviderKind::OpenAi, "net");
    let error = CogentError::from(error);

    assert!(matches!(
        error,
        CogentError::Remote(RemoteError {
            kind: RemoteErrorKind::Transport,
            remote_kind: RemoteKind::OpenAi,
            ..
        })
    ));
}
