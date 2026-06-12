//! Tests the `runtime::inference_runtime::embedding_read` module in `cogentlm`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::*;
use crate::engine::protocol::EmbedOptions;
use crate::engine::protocol::PoolingType;
use crate::error::Error;
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::request::GenerateRequest;
use crate::runtime::scheduler::SlotState;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn l2_normalize_unit_vector_is_idempotent() {
    let mut values = vec![1.0, 0.0, 0.0];
    l2_normalize(&mut values);
    assert!(approx_eq(values[0], 1.0));
    assert!(approx_eq(values[1], 0.0));
    assert!(approx_eq(values[2], 0.0));
}

#[test]
fn l2_normalize_scales_to_unit_length() {
    let mut values = vec![3.0, 4.0];
    l2_normalize(&mut values);
    let sum_sq: f32 = values.iter().map(|v| v * v).sum();
    assert!(approx_eq(sum_sq.sqrt(), 1.0));
    assert!(approx_eq(values[0], 0.6));
    assert!(approx_eq(values[1], 0.8));
}

#[test]
fn l2_normalize_zero_vector_is_left_alone() {
    let mut values = vec![0.0, 0.0, 0.0];
    l2_normalize(&mut values);
    for value in &values {
        assert_eq!(*value, 0.0);
    }
}

#[test]
fn apply_normalization_skips_rank_pooling() {
    let output = apply_normalization(vec![3.0, 4.0], PoolingType::Rank, true);
    assert!(!output.normalized);
    assert_eq!(output.pooling, PoolingType::Rank);
    assert_eq!(output.values, vec![3.0, 4.0]);
}

#[test]
fn apply_normalization_respects_normalize_request() {
    let output = apply_normalization(vec![3.0, 4.0], PoolingType::Mean, false);
    assert!(!output.normalized);
    assert_eq!(output.values, vec![3.0, 4.0]);

    let output = apply_normalization(vec![3.0, 4.0], PoolingType::Mean, true);
    assert!(output.normalized);
    assert!(approx_eq(output.values[0], 0.6));
    assert!(approx_eq(output.values[1], 0.8));
}

#[test]
fn slot_inputs_rejects_missing_slot() {
    let runtime = test_runtime(NativeRuntimeConfig::default());

    let error = slot_inputs(&runtime, 0).expect_err("missing slot");

    assert!(matches!(error, Error::RuntimeNotReady));
}

#[test]
fn slot_inputs_rejects_missing_request_and_embed_options() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.slot_scheduler.slots.push(SlotState::new(0));

    let error = slot_inputs(&runtime, 0).expect_err("missing request");
    assert!(matches!(
        error,
        Error::InvalidRequest(message) if message.contains("no request")
    ));

    let mut request = GenerateRequest::new(4, "ctx");
    request.prompt_tokens = vec![1];
    runtime.slot_scheduler.slots[0].request = Some(request);

    let error = slot_inputs(&runtime, 0).expect_err("missing embed options");
    assert!(matches!(
        error,
        Error::InvalidRequest(message) if message.contains("without embed options")
    ));
}

#[test]
fn read_pooled_embedding_rejects_negative_sequence_and_zero_dimensions() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());

    let error = runtime
        .read_pooled_embedding(-1)
        .expect_err("negative sequence id");
    assert!(matches!(
        error,
        Error::InvalidRequest(message) if message.contains("no sequence id")
    ));

    runtime.capabilities.embedding_dimensions = 0;
    runtime.capabilities.pooling_type = PoolingType::Mean;
    let error = runtime
        .read_pooled_embedding(0)
        .expect_err("zero dimensions");
    assert!(matches!(
        error,
        Error::UnsupportedOperation {
            operation: "embed",
            reason
        } if reason.contains("zero embedding dimensions")
    ));
}

#[test]
fn slot_inputs_returns_sequence_id_and_normalize_flag() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut slot = SlotState::new(0);
    let mut request = GenerateRequest::new(4, "ctx");
    request.embed_options = Some(EmbedOptions {
        normalize: false,
        context_key: Some("ctx".to_string()),
    });
    slot.seq_id = 9;
    slot.request = Some(request);
    runtime.slot_scheduler.slots.push(slot);

    assert_eq!(slot_inputs(&runtime, 0).expect("slot inputs"), (9, false));
}
