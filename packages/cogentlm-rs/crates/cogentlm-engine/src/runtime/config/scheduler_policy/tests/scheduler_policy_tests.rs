//! Unit tests for the parent module.

use super::super::*;

#[test]
fn scheduler_defaults_match_legacy_cpp_browser_runtime() {
    let policy = SchedulerPolicyConfig::default();

    assert_eq!(policy.mode, SchedulerPolicyMode::Balanced);
    assert_eq!(policy.decode_token_reserve, 1);
    assert!(!policy.enable_adaptive_prefill_chunking);
}

#[test]
fn scheduler_policy_choice_helpers_accept_binding_aliases() {
    assert_eq!(
        SchedulerPolicyMode::from_choice("latency"),
        Some(SchedulerPolicyMode::LatencyFirst)
    );
    assert_eq!(
        SchedulerPolicyMode::from_choice("throughput-first"),
        Some(SchedulerPolicyMode::ThroughputFirst)
    );
}
