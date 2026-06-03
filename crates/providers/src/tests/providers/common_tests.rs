//! Tests the `providers::common` module in `cogentlm-providers`.
//!
//! Covers shared provider request validation, provider option merging, usage
//! parsing, and provider body error normalization with deterministic JSON
//! fixtures and no HTTP calls.

use super::*;

#[test]
fn require_non_empty_field_accepts_non_blank_and_rejects_blank() {
    require_non_empty_field("model-a", "model", ProviderKind::Proxy).expect("non-empty");

    let err = require_non_empty_field("  ", "model", ProviderKind::Proxy)
        .expect_err("blank value should fail");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert!(err.message.contains("model"));
}

#[test]
fn insert_positive_u32_option_skips_none_inserts_positive_and_rejects_zero() {
    let mut body = serde_json::Map::new();
    insert_positive_u32_option(&mut body, "max_tokens", None, ProviderKind::Proxy)
        .expect("none is skipped");
    assert!(body.is_empty());

    insert_positive_u32_option(&mut body, "max_tokens", Some(12), ProviderKind::Proxy)
        .expect("positive value");
    assert_eq!(body["max_tokens"], 12);

    let err = insert_positive_u32_option(&mut body, "max_tokens", Some(0), ProviderKind::Proxy)
        .expect_err("zero should fail");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn insert_finite_f32_option_skips_none_inserts_finite_and_rejects_nan() {
    let mut body = serde_json::Map::new();
    insert_finite_f32_option(&mut body, "temperature", None, ProviderKind::Proxy)
        .expect("none is skipped");
    assert!(body.is_empty());

    insert_finite_f32_option(&mut body, "temperature", Some(0.75), ProviderKind::Proxy)
        .expect("finite value");
    assert_eq!(body["temperature"], serde_json::json!(0.75));

    let err = insert_finite_f32_option(
        &mut body,
        "temperature",
        Some(f32::NAN),
        ProviderKind::Proxy,
    )
    .expect_err("nan should fail");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn merge_provider_options_inserts_custom_options_and_rejects_typed_collisions() {
    let mut body = serde_json::Map::new();
    let provider_options = [("seed".to_string(), serde_json::json!(7))]
        .into_iter()
        .collect();

    merge_provider_options(
        &mut body,
        &provider_options,
        &["model"],
        ProviderKind::Proxy,
    )
    .expect("custom option");
    assert_eq!(body["seed"], 7);

    let provider_options = [("model".to_string(), serde_json::json!("other"))]
        .into_iter()
        .collect();
    let err = merge_provider_options(
        &mut body,
        &provider_options,
        &["model"],
        ProviderKind::Proxy,
    )
    .expect_err("typed field collision");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    assert!(err.message.contains("model"));
}

#[test]
fn optional_u32_handles_missing_valid_invalid_and_overflow_values() {
    assert_eq!(
        optional_u32(&serde_json::json!({}), "prompt_tokens", ProviderKind::Proxy)
            .expect("missing"),
        None
    );
    assert_eq!(
        optional_u32(
            &serde_json::json!({ "prompt_tokens": 12 }),
            "prompt_tokens",
            ProviderKind::Proxy
        )
        .expect("valid"),
        Some(12)
    );

    let err = optional_u32(
        &serde_json::json!({ "prompt_tokens": "12" }),
        "prompt_tokens",
        ProviderKind::Proxy,
    )
    .expect_err("string should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);

    let err = optional_u32(
        &serde_json::json!({ "prompt_tokens": u64::from(u32::MAX) + 1 }),
        "prompt_tokens",
        ProviderKind::Proxy,
    )
    .expect_err("overflow should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);
}

#[test]
fn token_usage_total_requires_both_sides_and_rejects_overflow() {
    assert_eq!(token_usage_total(Some(2), Some(3)), Some(5));
    assert_eq!(token_usage_total(None, Some(3)), None);
    assert_eq!(token_usage_total(Some(2), None), None);
    assert_eq!(token_usage_total(Some(u32::MAX), Some(1)), None);
}

#[test]
fn provider_body_error_uses_provider_codes_or_default_message() {
    let err = provider_body_error(
        serde_json::json!({
            "error": {
                "message": "quota exceeded",
                "code": "insufficient_quota"
            }
        }),
        ProviderKind::OpenAi,
        "fallback",
    );
    assert_eq!(err.kind, ProviderErrorKind::QuotaExceeded);
    assert_eq!(err.message, "quota exceeded");
    assert_eq!(err.code.as_deref(), Some("insufficient_quota"));
    assert!(err.raw.is_some());

    let err = provider_body_error(
        serde_json::json!({ "error": { "type": "unknown_type" } }),
        ProviderKind::Anthropic,
        "fallback",
    );
    assert_eq!(err.kind, ProviderErrorKind::Provider);
    assert_eq!(err.message, "fallback");
    assert_eq!(err.code.as_deref(), Some("unknown_type"));
}

#[test]
fn provider_response_error_sets_provider_kind_and_message() {
    let err = provider_response_error("bad response", ProviderKind::Proxy);

    assert_eq!(err.kind, ProviderErrorKind::Provider);
    assert_eq!(err.provider, ProviderKind::Proxy);
    assert_eq!(err.message, "bad response");
}
