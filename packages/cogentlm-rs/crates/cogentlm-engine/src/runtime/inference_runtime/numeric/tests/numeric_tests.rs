//! Unit tests for the parent module.

use super::super::*;

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
    assert_eq!(clamp_usize_to_u64(4), 4);
    assert_eq!(clamp_usize_to_u64(usize::MAX), u64::MAX);
    assert_eq!(ffi_arg_count_len(4).unwrap(), 4);
    let too_many_args = usize::try_from(i32::MAX)
        .ok()
        .and_then(|v| v.checked_add(1))
        .unwrap_or(usize::MAX);
    assert!(ffi_arg_count_len(too_many_args).is_err());
}

#[test]
fn token_piece_growth_capacity_returns_some_when_needed_exceeds_capacity() {
    assert_eq!(token_piece_growth_capacity(-128, 64), Some(128));
    assert_eq!(token_piece_growth_capacity(128, 64), Some(128));
    assert_eq!(token_piece_growth_capacity(-64, 64), None);
    assert_eq!(token_piece_growth_capacity(0, 64), None);
    assert_eq!(token_piece_growth_capacity(i32::MIN, 64), None);
}
