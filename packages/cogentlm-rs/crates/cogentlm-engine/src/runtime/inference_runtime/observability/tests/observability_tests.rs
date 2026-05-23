//! Unit tests for the parent module.

use super::super::increment_debug_counter;

#[test]
fn increment_debug_counter_saturates() {
    let mut counter = i32::MAX;

    increment_debug_counter(&mut counter);

    assert_eq!(counter, i32::MAX);
}
