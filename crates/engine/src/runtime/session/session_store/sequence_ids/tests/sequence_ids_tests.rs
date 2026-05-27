//! Unit tests for the parent module.

use super::super::*;

#[test]
fn sequence_capacity_clamps_to_representable_seq_ids() {
    assert_eq!(clamp_sequence_capacity(0), 1);
    assert_eq!(clamp_sequence_capacity(2), 2);
    assert_eq!(
        clamp_sequence_capacity(usize::MAX),
        max_representable_sequences()
    );
}
