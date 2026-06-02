//! Tests the `runtime::inference_runtime::multimodal` module in `cogentlm-engine`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::*;
use crate::runtime::request::{GenerateRequest, MultimodalPayload};
use crate::runtime::scheduler::SlotState;

fn slot_with_images(images: Vec<Vec<u8>>) -> SlotState {
    let mut slot = SlotState::new(0);
    let mut request = GenerateRequest::new(1, "ctx");
    request.multimodal = Some(MultimodalPayload {
        image_buffers: images,
    });
    slot.request = Some(request);
    slot
}

#[test]
fn flatten_image_buffers_concatenates_payloads_and_sizes() {
    let slot = slot_with_images(vec![vec![1, 2], vec![3], Vec::new(), vec![4, 5, 6]]);

    let (bytes, sizes) = flatten_image_buffers(&slot).expect("flattened images");

    assert_eq!(bytes, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(sizes, vec![2, 1, 0, 3]);
}

#[test]
fn flatten_image_buffers_requires_multimodal_payload() {
    let slot = SlotState::new(0);

    assert!(flatten_image_buffers(&slot).is_none());
}

#[test]
fn clear_multimodal_payload_only_drops_media_state() {
    let mut slot = slot_with_images(vec![vec![1, 2, 3]]);
    slot.buffered_output_text = "pending".to_string();

    clear_multimodal_payload(&mut slot);

    let request = slot.request().expect("request remains attached");
    assert!(request.multimodal.is_none());
    assert_eq!(slot.buffered_output_text, "pending");
}

#[test]
fn clear_multimodal_payload_allows_missing_request() {
    let mut slot = SlotState::new(0);

    clear_multimodal_payload(&mut slot);

    assert!(slot.request().is_none());
}
