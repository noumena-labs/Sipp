//! Tests the `runtime::inference_runtime::multimodal` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::*;
use crate::native_bridge::{NativeRuntimeHandle, SamplerHandle};
use crate::runtime::request::RequestQueue;
use crate::runtime::request::{GenerateRequest, MultimodalPayload};
use crate::runtime::scheduler::SlotState;

fn slot_with_media(media_buffers: Vec<Vec<u8>>) -> SlotState {
    let mut slot = SlotState::new(0);
    let mut request = GenerateRequest::new(1, "ctx");
    request.multimodal = Some(MultimodalPayload { media_buffers });
    slot.request = Some(request);
    slot
}

#[test]
fn media_parts_borrows_one_payload_without_copying() {
    let media = vec![vec![1, 2, 3]];

    let (bytes, sizes) = media_parts(&media).expect("media parts");

    assert!(matches!(bytes, Cow::Borrowed(_)));
    assert_eq!(bytes.as_ref(), [1, 2, 3]);
    assert_eq!(sizes, vec![3]);
}

#[test]
fn media_parts_flattens_multiple_payloads() {
    let media = vec![vec![1, 2], vec![3], vec![4, 5, 6]];

    let (bytes, sizes) = media_parts(&media).expect("media parts");

    assert_eq!(bytes, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(sizes, vec![2, 1, 3]);
}

#[test]
fn run_multimodal_prefill_rejects_missing_prerequisites_before_native_work() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut queue = RequestQueue::new();
    let mut scratch = Vec::new();

    let mut missing_request = SlotState::new(0);
    missing_request.set_sampler(SamplerHandle::empty_for_tests());
    assert!(!run_multimodal_prefill(
        &mut runtime,
        4,
        &mut queue,
        &mut missing_request,
        &mut scratch
    ));

    let mut missing_sampler = slot_with_media(vec![vec![1]]);
    assert!(!run_multimodal_prefill(
        &mut runtime,
        4,
        &mut queue,
        &mut missing_sampler,
        &mut scratch
    ));
    assert!(missing_sampler
        .request()
        .and_then(|request| request.multimodal.as_ref())
        .is_some());
}

#[test]
fn run_multimodal_prefill_clears_payload_when_mtmd_context_is_not_ready() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut queue = RequestQueue::new();
    let mut scratch = Vec::new();
    let mut slot = slot_with_media(vec![vec![1, 2, 3]]);
    slot.seq_id = 0;
    slot.set_sampler(SamplerHandle::empty_for_tests());

    assert!(!run_multimodal_prefill(
        &mut runtime,
        4,
        &mut queue,
        &mut slot,
        &mut scratch
    ));

    assert!(slot
        .request()
        .and_then(|request| request.multimodal.as_ref())
        .is_none());
}
