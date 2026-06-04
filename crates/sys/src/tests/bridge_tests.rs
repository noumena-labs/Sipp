//! Tests the `bridge` module in `cogentlm-sys`.
//!
//! Covers model-free CXX bridge behavior for native batch allocation,
//! token storage, bounds handling, and logit flag helpers without loading a model.

use super::bridge::*;

#[test]
fn new_batch_is_empty_and_safe_before_allocation() {
    let batch = make_native_batch();
    let batch = batch.as_ref().expect("native batch");

    assert_eq!(batch.n_tokens(), 0);
    for index in [-1, 0, 1] {
        assert_eq!(batch.token(index), 0);
        assert_eq!(batch.pos(index), 0);
        assert_eq!(batch.seq_id(index), 0);
        assert!(!batch.logits(index));
    }
}

#[test]
fn ensure_capacity_rejects_non_positive_values() {
    let mut batch = make_native_batch();

    let token_error = batch
        .pin_mut()
        .ensure_capacity(0, 1)
        .expect_err("zero token capacity should fail");
    assert!(
        token_error
            .to_string()
            .contains("llama batch capacity must be positive"),
        "{token_error}"
    );

    let sequence_error = batch
        .pin_mut()
        .ensure_capacity(1, 0)
        .expect_err("zero sequence capacity should fail");
    assert!(
        sequence_error
            .to_string()
            .contains("llama batch capacity must be positive"),
        "{sequence_error}"
    );
}

#[test]
fn add_token_populates_batch_arrays_and_clamps_capacity() {
    let mut batch = make_native_batch();
    batch.pin_mut().ensure_capacity(2, 1).expect("allocate");

    assert!(batch.pin_mut().add_token(42, 7, 3, true));
    assert!(batch.pin_mut().add_token(-5, 8, 4, false));
    assert!(!batch.pin_mut().add_token(99, 9, 5, true));

    let batch = batch.as_ref().expect("native batch");
    assert_eq!(batch.n_tokens(), 2);
    assert_eq!(batch.token(0), 42);
    assert_eq!(batch.pos(0), 7);
    assert_eq!(batch.seq_id(0), 3);
    assert!(batch.logits(0));
    assert_eq!(batch.token(1), -5);
    assert_eq!(batch.pos(1), 8);
    assert_eq!(batch.seq_id(1), 4);
    assert!(!batch.logits(1));
    assert_eq!(batch.token(2), 0);
    assert_eq!(batch.pos(-1), 0);
    assert_eq!(batch.seq_id(2), 0);
    assert!(!batch.logits(2));
}

#[test]
fn ensure_capacity_reuses_existing_allocation_and_resets_tokens() {
    let mut batch = make_native_batch();
    batch.pin_mut().ensure_capacity(2, 1).expect("allocate");
    assert!(batch.pin_mut().add_token(10, 0, 0, false));
    assert_eq!(batch.as_ref().expect("native batch").n_tokens(), 1);

    batch.pin_mut().ensure_capacity(2, 1).expect("reuse");
    assert_eq!(batch.as_ref().expect("native batch").n_tokens(), 0);

    batch
        .pin_mut()
        .ensure_capacity(1, 1)
        .expect("reuse larger existing allocation");
    assert_eq!(batch.as_ref().expect("native batch").n_tokens(), 0);
    assert!(batch.pin_mut().add_token(11, 0, 0, false));
    assert!(batch.pin_mut().add_token(12, 1, 0, false));
    assert!(!batch.pin_mut().add_token(13, 2, 0, false));
}

#[test]
fn logit_helpers_clear_and_mark_only_last_token() {
    let mut batch = make_native_batch();
    batch.pin_mut().ensure_capacity(3, 1).expect("allocate");
    assert!(batch.pin_mut().add_token(1, 0, 0, true));
    assert!(batch.pin_mut().add_token(2, 1, 0, true));

    batch.pin_mut().clear_logits();
    let native = batch.as_ref().expect("native batch");
    assert!(!native.logits(0));
    assert!(!native.logits(1));

    batch.pin_mut().set_last_logits();
    let native = batch.as_ref().expect("native batch");
    assert!(!native.logits(0));
    assert!(native.logits(1));

    batch.pin_mut().reset();
    batch.pin_mut().set_last_logits();
    assert_eq!(batch.as_ref().expect("native batch").n_tokens(), 0);
}
