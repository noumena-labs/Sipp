//! Tests the `runtime::scheduler::slot_scheduler::metrics` module in `cogentlm`.
//!
//! Covers scheduler planning, budget accounting, slot state, and flow decisions with deterministic in-memory fixtures.

use crate::runtime::numeric::saturating_usize_to_i32;

#[test]
fn usize_metrics_saturate_at_i32_max() {
    assert_eq!(saturating_usize_to_i32(i32::MAX as usize + 1), i32::MAX);
}
