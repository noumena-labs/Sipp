//! Unit tests for the parent module.

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
fn provider_error_kind_from_code_maps_gateway_stable_codes() {
    for (code, kind) in [
        ("authentication", ProviderErrorKind::Authentication),
        ("authorization", ProviderErrorKind::Authorization),
        ("invalid_request", ProviderErrorKind::InvalidRequest),
        ("unsupported_feature", ProviderErrorKind::UnsupportedFeature),
        ("model_not_found", ProviderErrorKind::ModelNotFound),
        ("overloaded", ProviderErrorKind::Overloaded),
        ("quota_exceeded", ProviderErrorKind::QuotaExceeded),
        ("rate_limited", ProviderErrorKind::RateLimited),
        ("timeout", ProviderErrorKind::Timeout),
        ("transport", ProviderErrorKind::Transport),
    ] {
        assert_eq!(
            provider_error_kind_from_code(Some(code)),
            Some(kind),
            "{code}"
        );
    }
}

#[test]
fn provider_error_debug_redacts_provider_payloads() {
    let error = ProviderError {
        kind: ProviderErrorKind::Authentication,
        provider: ProviderKind::OpenAiCompatible,
        status: Some(401),
        code: Some("provider-secret-code".to_string()),
        message: "provider rejected provider-secret-token".to_string(),
        retry_after: Some(std::time::Duration::from_millis(500)),
        request_id: Some("req-provider-secret-token".to_string()),
        raw: Some(Box::new(serde_json::json!({
            "error": {
                "message": "provider-secret-token",
                "code": "provider-secret-code"
            }
        }))),
    };

    let debug = format!("{error:?}");

    assert!(!debug.contains("provider-secret"));
    assert!(debug.contains("[redacted]"));
}

#[test]
fn provider_error_display_does_not_expose_provider_payloads() {
    let error = ProviderError {
        kind: ProviderErrorKind::Authentication,
        provider: ProviderKind::OpenAiCompatible,
        status: Some(401),
        code: Some("provider-secret-code".to_string()),
        message: "provider rejected provider-secret-token".to_string(),
        retry_after: Some(std::time::Duration::from_millis(500)),
        request_id: Some("req-provider-secret-token".to_string()),
        raw: Some(Box::new(serde_json::json!({
            "error": {
                "message": "provider-secret-token",
                "code": "provider-secret-code"
            }
        }))),
    };

    let display = error.to_string();

    assert!(!display.contains("provider-secret"));
    assert!(display.contains("OpenAiCompatible provider error (Authentication)"));
}
