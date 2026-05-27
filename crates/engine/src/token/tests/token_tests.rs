//! Unit tests for the parent module.

use super::super::*;

#[test]
fn next_ffi_capacity_rejects_non_growth_and_abs_overflow() {
    assert_eq!(next_ffi_capacity(-128, 64), Some(128));
    assert_eq!(next_ffi_capacity(128, 64), Some(128));
    assert_eq!(next_ffi_capacity(-64, 64), None);
    assert_eq!(next_ffi_capacity(0, 64), None);
    assert_eq!(next_ffi_capacity(i32::MIN, 64), None);
}

#[test]
fn initial_token_capacity_checks_padding_and_i32_bounds() {
    assert_eq!(initial_token_capacity(0), Some(8));
    assert_eq!(initial_token_capacity(1), Some(9));
    assert_eq!(
        initial_token_capacity((i32::MAX as usize) - 8),
        Some(i32::MAX)
    );
    assert_eq!(initial_token_capacity((i32::MAX as usize) - 7), None);
    assert_eq!(initial_token_capacity(usize::MAX), None);
}
