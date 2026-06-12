//! Tests the `runtime::config::scheduler_policy` module in
//! `cogentlm`.
//!
//! Covers scheduler policy serde defaults and effective budget clamping with
//! pure value assertions.

use serde_json::json;

use super::*;

#[test]
fn scheduler_policy_mode_uses_snake_case_wire_names() {
    assert_eq!(
        SchedulerPolicyMode::default(),
        SchedulerPolicyMode::Balanced
    );
    assert_eq!(
        serde_json::to_value(SchedulerPolicyMode::LatencyFirst).expect("mode"),
        "latency_first"
    );
    assert_eq!(
        serde_json::from_value::<SchedulerPolicyMode>(json!("throughput_first")).expect("mode"),
        SchedulerPolicyMode::ThroughputFirst
    );
}

#[test]
fn scheduler_policy_config_deserializes_defaults_and_rejects_unknown_fields() {
    let default_config = SchedulerPolicyConfig::default();
    assert_eq!(default_config.mode, SchedulerPolicyMode::Balanced);
    assert_eq!(default_config.decode_token_reserve, 1);
    assert!(!default_config.enable_adaptive_prefill_chunking);

    let config: SchedulerPolicyConfig =
        serde_json::from_str(r#"{"mode":"latency_first"}"#).expect("policy");

    assert_eq!(config.mode, SchedulerPolicyMode::LatencyFirst);
    assert_eq!(config.decode_token_reserve, 1);
    assert!(!config.enable_adaptive_prefill_chunking);

    let error = serde_json::from_str::<SchedulerPolicyConfig>(r#"{"unknown":true}"#)
        .expect_err("unknown field");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn scheduler_tick_budget_effective_values_clamp_to_total_budget() {
    let default_budget = SchedulerTickBudget::default();
    assert_eq!(default_budget.total_token_budget, 0);
    assert_eq!(default_budget.reserved_decode_tokens, 0);
    assert_eq!(default_budget.reserved_prefill_tokens, 0);
    assert!(!default_budget.decode_first);
    assert_eq!(default_budget.effective_decode_budget(), 0);
    assert_eq!(default_budget.effective_prefill_budget(), 0);

    let budget = SchedulerTickBudget {
        total_token_budget: 4,
        reserved_decode_tokens: 10,
        reserved_prefill_tokens: 10,
        decode_first: true,
    };
    assert_eq!(budget.effective_decode_budget(), 4);
    assert_eq!(budget.effective_prefill_budget(), 0);

    let budget = SchedulerTickBudget {
        total_token_budget: 10,
        reserved_decode_tokens: -1,
        reserved_prefill_tokens: 5,
        decode_first: false,
    };
    assert_eq!(budget.effective_decode_budget(), 0);
    assert_eq!(budget.effective_prefill_budget(), 5);

    let budget = SchedulerTickBudget {
        total_token_budget: 10,
        reserved_decode_tokens: 4,
        reserved_prefill_tokens: 10,
        decode_first: true,
    };
    assert_eq!(budget.effective_decode_budget(), 4);
    assert_eq!(budget.effective_prefill_budget(), 6);
}
