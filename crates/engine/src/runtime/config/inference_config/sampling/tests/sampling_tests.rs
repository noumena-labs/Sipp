//! Unit tests for the parent module.

use super::super::should_merge_sampling_override;

#[test]
fn sampling_override_ignores_nulls_and_empty_arrays() {
    let mut base = serde_json::json!({
        "top_k": 40,
        "samplers": ["top_k"],
        "backend_sampling": true
    });
    let override_value = serde_json::json!({
        "top_k": 12,
        "samplers": [],
        "backend_sampling": null
    });

    super::super::merge_sampling_override_json(&mut base, override_value);

    assert_eq!(base["top_k"], 12);
    assert_eq!(base["samplers"], serde_json::json!(["top_k"]));
    assert_eq!(base["backend_sampling"], true);
}

#[test]
fn sampling_override_merge_policy_keeps_meaningful_values_only() {
    assert!(!should_merge_sampling_override(&serde_json::Value::Null));
    assert!(!should_merge_sampling_override(&serde_json::json!([])));
    assert!(should_merge_sampling_override(&serde_json::json!([1])));
    assert!(should_merge_sampling_override(&serde_json::json!(false)));
    assert!(should_merge_sampling_override(&serde_json::json!(0)));
    assert!(should_merge_sampling_override(&serde_json::json!("")));
}
