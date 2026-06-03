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
