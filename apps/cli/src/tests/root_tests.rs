//! Tests the `crate root` module in `sipp-cli`.
//!
//! Covers CLI parsing, configuration mapping, stats rendering, and command behavior without running model-backed inference unless marked as an external smoke test.

use super::*;

#[test]
fn cli_positive_i32_accepts_positive_i32_range_values() {
    assert_eq!(cli_positive_i32(1, "value").unwrap(), 1);
    assert_eq!(
        cli_positive_i32(i32::MAX as u32, "value").unwrap(),
        i32::MAX
    );
}

#[test]
fn cli_positive_i32_rejects_zero_and_overflow() {
    assert!(cli_positive_i32(0, "value")
        .unwrap_err()
        .to_string()
        .contains("must be positive"));
    assert!(cli_positive_i32(i32::MAX as u32 + 1, "value")
        .unwrap_err()
        .to_string()
        .contains("signed 32-bit"));
}
