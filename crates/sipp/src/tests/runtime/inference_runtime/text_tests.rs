//! Tests the `runtime::inference_runtime::text` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::*;
use crate::runtime::request::GenerateRequest;
use crate::runtime::scheduler::{SlotPhase, SlotState};

fn slot_with_stop(stops: Vec<&str>) -> SlotState {
    let mut slot = SlotState::new(0);
    let mut request = GenerateRequest::new(1, "ctx");
    request.stop = stops.into_iter().map(str::to_string).collect();
    slot.request = Some(request);
    slot
}

#[test]
fn detects_incomplete_utf8_tail() {
    assert_eq!(incomplete_utf8_tail_length(b"a"), 0);
    assert_eq!(incomplete_utf8_tail_length(&[0xE2, 0x82]), 2);
    assert_eq!(incomplete_utf8_tail_length(&[0xE2, 0x82, 0xAC]), 0);
}

#[test]
fn earliest_stop_index_split_finds_cross_boundary_matches() {
    let stops = vec!["🌍!".to_string(), "stop".to_string(), "\n".to_string()];

    assert_eq!(
        earliest_stop_index_split("hello st", "op world", &stops),
        Some(6)
    );
    assert_eq!(
        earliest_stop_index_split("hello, 🌍", "! world", &stops),
        Some(7)
    );
    assert_eq!(
        earliest_stop_index_split("hello", "world stop", &stops),
        Some(11)
    );
    assert_eq!(earliest_stop_index_split("hello", "world", &stops), None);
}

#[test]
fn apply_stop_sequences_truncates_output_and_clears_buffers() {
    let mut slot = slot_with_stop(vec!["stop"]);
    slot.output_text = "hello ".to_string();
    slot.buffered_output_text = "stop and more".to_string();
    slot.pending_emission_text = "pending".to_string();
    slot.pending_utf8_bytes = vec![0xE2, 0x82];

    assert!(apply_stop_sequences_to_slot(&mut slot));

    assert_eq!(slot.output_text, "hello ");
    assert!(slot.buffered_output_text.is_empty());
    assert!(slot.pending_emission_text.is_empty());
    assert!(slot.pending_utf8_bytes.is_empty());
    assert_eq!(slot.phase, SlotPhase::Completed);
}

#[test]
fn apply_stop_sequences_holds_possible_cross_boundary_suffix() {
    let mut slot = slot_with_stop(vec!["abcd"]);
    slot.buffered_output_text = "xxab".to_string();

    assert!(!apply_stop_sequences_to_slot(&mut slot));

    assert_eq!(slot.buffered_output_text, "x");
    assert_eq!(slot.pending_emission_text, "xab");
}

#[test]
fn truncate_to_char_boundary_never_splits_utf8() {
    let mut value = "aé日".to_string();

    truncate_to_char_boundary(&mut value, 3);

    assert_eq!(value, "aé");
}

#[test]
fn flush_pending_utf8_moves_pending_text_and_lossy_bytes_to_buffer() {
    let mut slot = SlotState::new(0);
    slot.pending_emission_text = "ok".to_string();
    slot.pending_utf8_bytes = vec![0xE2, 0x82];

    flush_pending_utf8(&mut slot);

    assert!(slot.pending_emission_text.is_empty());
    assert!(slot.pending_utf8_bytes.is_empty());
    assert_eq!(slot.buffered_output_text, "ok�");
}
