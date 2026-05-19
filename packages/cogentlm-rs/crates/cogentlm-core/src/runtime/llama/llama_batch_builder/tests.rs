//! Unit tests for the parent module.

use super::*;

#[test]
fn ensure_capacity_allocates_and_reuses_matching_batch() {
    let mut builder = LlamaBatchBuilder::default();
    builder.ensure_capacity(2, 1).expect("allocate");
    assert!(builder.is_allocated());
    assert_eq!(builder.capacity_tokens(), 2);
    assert!(builder.add_token(10, 0, 0, false));
    assert_eq!(builder.raw().n_tokens, 1);

    builder.ensure_capacity(2, 1).expect("reuse");
    assert_eq!(builder.raw().n_tokens, 0);
    assert_eq!(builder.capacity_tokens(), 2);

    builder.ensure_capacity(1, 1).expect("reuse-shrunk");
    assert_eq!(builder.raw().n_tokens, 0);
    assert_eq!(builder.capacity_tokens(), 2);
}

#[test]
fn add_token_populates_batch_arrays_and_clamps_capacity() {
    let mut builder = LlamaBatchBuilder::default();
    builder.ensure_capacity(1, 1).expect("allocate");

    assert!(builder.add_token(42, 7, 3, true));
    assert!(!builder.add_token(43, 8, 3, false));
    assert_eq!(builder.raw().n_tokens, 1);
    unsafe {
        assert_eq!(*builder.raw().token, 42);
        assert_eq!(*builder.raw().pos, 7);
        assert_eq!(*builder.raw().n_seq_id, 1);
        assert_eq!(**builder.raw().seq_id, 3);
        assert_eq!(*builder.raw().logits, 1);
    }
}

#[test]
fn add_token_rejects_invalid_signed_token_count() {
    let mut builder = LlamaBatchBuilder::default();
    builder.ensure_capacity(1, 1).expect("allocate");

    builder.raw_mut().n_tokens = -1;

    assert!(!builder.add_token(42, 7, 3, true));
    assert_eq!(builder.raw().n_tokens, -1);
}
