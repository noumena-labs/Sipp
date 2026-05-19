//! Unit tests for the parent module.

use super::*;

#[test]
fn scheduler_defaults_match_legacy_cpp_browser_runtime() {
    let policy = SchedulerPolicyConfig::default();

    assert_eq!(policy.mode, SchedulerPolicyMode::Balanced);
    assert_eq!(policy.decode_token_reserve, 1);
    assert!(!policy.enable_adaptive_prefill_chunking);
}
