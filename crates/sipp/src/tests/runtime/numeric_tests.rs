//! Tests the `runtime::numeric` module in `sipp`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use super::*;

#[test]
fn system_time_unix_ms_saturates_and_handles_pre_epoch() {
    assert_eq!(
        system_time_unix_ms(UNIX_EPOCH - Duration::from_millis(1)),
        0
    );
    assert_eq!(
        system_time_unix_ms(UNIX_EPOCH + Duration::from_millis(42)),
        42
    );
}

#[test]
fn positive_fair_share_clamps_divisor_and_result() {
    assert_eq!(positive_fair_share_i32(8, 2), 4);
    assert_eq!(positive_fair_share_i32(1, 2), 1);
    assert_eq!(positive_fair_share_i32(8, 0), 8);
    assert_eq!(positive_fair_share_i32(0, 4), 1);
}

#[test]
fn sign_clamps_preserve_nonnegative_and_positive_bounds() {
    assert_eq!(nonnegative_i32(-1), 0);
    assert_eq!(nonnegative_i32(2), 2);
    assert_eq!(positive_i32(0), 1);
    assert_eq!(positive_i32(2), 2);
    assert_eq!(positive_usize(0), 1);
    assert_eq!(positive_usize(2), 2);
    assert_eq!(positive_i32_to_usize(0), None);
    assert_eq!(positive_i32_to_usize(4), Some(4));
}
