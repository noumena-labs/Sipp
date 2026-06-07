//! Tests the `error` module in `cogentlm-client`.
//!
//! Covers gateway-to-client remote error classification and stable diagnostic
//! labels using synthetic gateway errors without remote transport calls.

#[cfg(feature = "remote")]
use std::time::Duration;

#[cfg(feature = "remote")]
use cogentlm_remote::{GatewayError, GatewayErrorKind};
#[cfg(feature = "remote")]
use serde_json::json;

#[cfg(feature = "remote")]
use super::*;

#[cfg(feature = "remote")]
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
        (RemoteErrorKind::ServerRestarting, "server_restarting"),
        (RemoteErrorKind::Transport, "transport"),
        (RemoteErrorKind::Remote, "remote"),
    ];

    for (kind, label) in cases {
        assert_eq!(kind.as_str(), label);
    }
}

#[cfg(feature = "remote")]
#[test]
fn remote_error_new_sets_required_fields_and_display() {
    let error = RemoteError::new(RemoteErrorKind::Timeout, "slow");

    assert_eq!(error.kind, RemoteErrorKind::Timeout);
    assert_eq!(error.message, "slow");
    assert!(error.status.is_none());
    assert!(error.code.is_none());
    assert!(error.retry_after.is_none());
    assert!(error.request_id.is_none());
    assert!(error.raw.is_none());
    assert_eq!(error.to_string(), "remote gateway error (timeout): slow");
}

#[cfg(feature = "remote")]
#[test]
fn gateway_error_kind_conversion_covers_all_variants() {
    let cases = [
        (
            GatewayErrorKind::Authentication,
            RemoteErrorKind::Authentication,
        ),
        (
            GatewayErrorKind::Authorization,
            RemoteErrorKind::Authorization,
        ),
        (GatewayErrorKind::RateLimited, RemoteErrorKind::RateLimited),
        (
            GatewayErrorKind::QuotaExceeded,
            RemoteErrorKind::QuotaExceeded,
        ),
        (
            GatewayErrorKind::InvalidRequest,
            RemoteErrorKind::InvalidRequest,
        ),
        (
            GatewayErrorKind::UnsupportedFeature,
            RemoteErrorKind::UnsupportedFeature,
        ),
        (
            GatewayErrorKind::ModelNotFound,
            RemoteErrorKind::ModelNotFound,
        ),
        (GatewayErrorKind::Timeout, RemoteErrorKind::Timeout),
        (GatewayErrorKind::Overloaded, RemoteErrorKind::Overloaded),
        (
            GatewayErrorKind::ServerRestarting,
            RemoteErrorKind::ServerRestarting,
        ),
        (GatewayErrorKind::Transport, RemoteErrorKind::Transport),
        (GatewayErrorKind::Gateway, RemoteErrorKind::Remote),
    ];

    for (gateway_kind, remote_kind) in cases {
        let error = GatewayError::new(gateway_kind, "gateway error");
        let remote = RemoteError::from(error);

        assert_eq!(remote.kind, remote_kind);
        assert_eq!(remote.message, "gateway error");
    }
}

#[cfg(feature = "remote")]
#[test]
fn gateway_metadata_is_preserved_when_converted() {
    let mut error = GatewayError::new(GatewayErrorKind::RateLimited, "limited");
    error.status = Some(429);
    error.code = Some("rate_limit_error".to_string());
    error.retry_after = Some(Duration::from_secs(2));
    error.request_id = Some("req-1".to_string());
    error.raw = Some(Box::new(json!({ "error": "limited" })));

    let remote = RemoteError::from(error);

    assert_eq!(remote.kind, RemoteErrorKind::RateLimited);
    assert_eq!(remote.status, Some(429));
    assert_eq!(remote.code.as_deref(), Some("rate_limit_error"));
    assert_eq!(remote.retry_after, Some(Duration::from_secs(2)));
    assert_eq!(remote.request_id.as_deref(), Some("req-1"));
    assert_eq!(remote.raw.as_deref(), Some(&json!({ "error": "limited" })));
}

#[cfg(feature = "remote")]
#[test]
fn gateway_error_converts_into_cogent_remote_error() {
    let error = GatewayError::new(GatewayErrorKind::Transport, "net");
    let error = CogentError::from(error);

    assert!(matches!(
        error,
        CogentError::Remote(RemoteError {
            kind: RemoteErrorKind::Transport,
            ..
        })
    ));
}
