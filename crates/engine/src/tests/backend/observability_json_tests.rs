//! Tests the `backend::observability_json` module in `cogentlm-engine`.
//!
//! Covers JSON helper extraction and fallback behavior with deterministic
//! `serde_json::Value` fixtures and no backend FFI calls.

use serde_json::json;

use super::*;

#[test]
fn json_extractors_return_typed_values_only() {
    let value = json!({
        "items": [{ "name": "cpu" }],
        "enabled": true,
        "name": "runtime",
        "bytes": 42,
        "wrong": "42"
    });

    assert_eq!(json_array(&value, "items").map(<[_]>::len), Some(1));
    assert_eq!(json_bool(&value, "enabled"), Some(true));
    assert_eq!(json_str(&value, "name"), Some("runtime"));
    assert_eq!(json_u64(&value, "bytes"), Some(42));
    assert_eq!(json_u64(&value, "wrong"), None);
    assert_eq!(json_array(&value, "missing"), None);
    assert_eq!(json_bool(&value, "name"), None);
    assert_eq!(json_str(&value, "enabled"), None);
}

#[test]
fn json_string_helpers_preserve_order_and_fallbacks() {
    let value = json!({
        "devices": [
            { "name": "gpu-0" },
            { "name": 7 },
            { "other": "skip" },
            { "name": "cpu" }
        ]
    });

    assert_eq!(json_string_or(&value, "missing", "fallback"), "fallback");
    assert_eq!(
        json_array_strings(&value, "devices", "name"),
        ["gpu-0", "cpu"]
    );
    assert!(json_array_strings(&value, "missing", "name").is_empty());
}
