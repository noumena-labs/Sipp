//! Tests the `runtime::llama::llama_batch_builder` module in `sipp`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use super::*;

#[test]
fn ensure_capacity_allocates_and_reuses_matching_batch() {
    let mut builder = LlamaBatchBuilder::default();
    builder.ensure_capacity(2, 1).expect("allocate");
    assert!(builder.add_token(10, 0, 0, false));
    assert_eq!(builder.n_tokens(), 1);

    builder.ensure_capacity(2, 1).expect("reuse");
    assert_eq!(builder.n_tokens(), 0);

    builder.ensure_capacity(1, 1).expect("reuse-shrunk");
    assert_eq!(builder.n_tokens(), 0);
    assert!(builder.add_token(11, 0, 0, false));
    assert!(builder.add_token(12, 1, 0, false));
    assert!(!builder.add_token(13, 2, 0, false));
}

#[test]
fn add_token_populates_batch_arrays_and_clamps_capacity() {
    let mut builder = LlamaBatchBuilder::default();
    builder.ensure_capacity(1, 1).expect("allocate");

    assert!(builder.add_token(42, 7, 3, true));
    assert!(!builder.add_token(43, 8, 3, false));
    assert_eq!(builder.n_tokens(), 1);
    assert_eq!(builder.token(0), 42);
    assert_eq!(builder.pos(0), 7);
    assert_eq!(builder.seq_id(0), 3);
    assert!(builder.logits(0));
}

#[test]
fn add_token_rejects_capacity_exhaustion() {
    let mut builder = LlamaBatchBuilder::default();
    builder.ensure_capacity(1, 1).expect("allocate");

    assert!(builder.add_token(42, 7, 3, true));
    assert!(!builder.add_token(43, 8, 3, true));
    assert_eq!(builder.n_tokens(), 1);
}
