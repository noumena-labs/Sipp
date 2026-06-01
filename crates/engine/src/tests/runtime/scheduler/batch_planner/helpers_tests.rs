//! Unit tests for the parent module.

use super::*;
use crate::runtime::numeric::saturating_u32_to_i32;

#[test]
fn conversions_saturate_scheduler_counts() {
    assert_eq!(saturating_usize_to_i32(i32::MAX as usize + 1), i32::MAX);
    assert_eq!(saturating_u32_to_i32(u32::MAX), i32::MAX);
    assert_eq!(positive_i32_to_usize(0), None);
    assert_eq!(positive_i32_to_usize(4), Some(4));
    assert!(token_limit_reached(4, 4));
    assert!(!token_limit_reached(usize::MAX, -1));
}
