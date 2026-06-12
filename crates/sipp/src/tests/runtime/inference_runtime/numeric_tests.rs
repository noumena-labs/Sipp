//! Tests the `runtime::inference_runtime::numeric` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::*;

#[test]
fn conversion_helpers_clamp_and_validate_boundaries() {
    assert_eq!(clamp_usize_to_i32(usize::MAX), i32::MAX);
    assert_eq!(positive_i32_to_usize(-8), 1);
    assert_eq!(positive_i32_to_usize(0), 1);
    assert_eq!(positive_i32_to_usize(4), 4);
    assert_eq!(nonnegative_i32_to_usize(-8), 0);
    assert_eq!(nonnegative_i32_to_usize(4), 4);
    assert_eq!(nonnegative_i32_to_usize_opt(-8), None);
    assert_eq!(nonnegative_i32_to_usize_opt(4), Some(4));
    assert_eq!(usize_to_i32(4), Some(4));
    assert_eq!(usize_to_i32(usize::MAX), None);
}
