//! Tests the `runtime::scheduler::slot_scheduler::flow` module in `sipp`.
//!
//! Covers scheduler planning, budget accounting, slot state, and flow decisions with deterministic in-memory fixtures.

use std::sync::Arc;

use crate::runtime::config::KvReuseMode;
use crate::runtime::request::{
    token_byte_ring, GenerateRequest, GenerateRequestId, GenerateResponseStatus, RequestQueue,
    ResponseOutput,
};
use crate::runtime::scheduler::{SlotExecutionPlan, SlotPhase};
use crate::runtime::session::{CacheCandidate, KvCacheAdmission, KvCacheManager, SequenceMirror};

use super::*;

fn request(id: GenerateRequestId, context_key: &str) -> GenerateRequest {
    let mut request = GenerateRequest::new(id, context_key);
    request.prompt_tokens = vec![1, 2, 3];
    request
}

fn admission(seq_id: i32) -> KvCacheAdmission {
    KvCacheAdmission {
        seq_id,
        generation: 1,
        mirror: SequenceMirror::default(),
        candidate: CacheCandidate::None,
    }
}

fn admit_one(
    scheduler: &mut SlotScheduler,
    queue: &mut RequestQueue,
    kv_cache: &mut KvCacheManager,
) -> Option<usize> {
    scheduler.admit_pending_requests(queue, kv_cache, KvReuseMode::LiveSlotPrefix, |_| {
        Some(SlotExecutionPlan::default())
    })
}

#[test]
fn resize_resets_non_idle_slots_and_preserves_slot_ids() {
    let mut scheduler = SlotScheduler::default();
    let mut kv_cache = KvCacheManager::new(2);
    scheduler.resize(2, &mut kv_cache);
    scheduler.slots[0].attach_request(request(1, "a"), admission(0));
    scheduler.slots[0].phase = SlotPhase::Decode;

    scheduler.resize(2, &mut kv_cache);

    assert_eq!(scheduler.slots[0].slot_id, 0);
    assert_eq!(scheduler.slots[0].phase, SlotPhase::Idle);
    assert_eq!(scheduler.slots[1].slot_id, 1);
}

#[test]
fn selects_decode_ready_slots_without_buffered_text() {
    let mut scheduler = SlotScheduler::default();
    let mut kv_cache = KvCacheManager::new(3);
    scheduler.resize(3, &mut kv_cache);
    scheduler.slots[0].attach_request(request(1, "a"), admission(0));
    scheduler.slots[0].phase = SlotPhase::Decode;
    scheduler.slots[0].generated_tokens.push(10);
    scheduler.slots[1].attach_request(request(2, "b"), admission(1));
    scheduler.slots[1].phase = SlotPhase::Decode;
    scheduler.slots[1].generated_tokens.push(11);
    scheduler.slots[1].buffered_output_text = "wait".to_string();

    let mut ready = Vec::new();
    scheduler.select_decode_ready_slots_into(&mut ready);
    assert_eq!(ready, vec![0]);
}

#[test]
fn selects_prefill_slots_with_remaining_prompt_tokens() {
    let mut scheduler = SlotScheduler::default();
    let mut kv_cache = KvCacheManager::new(2);
    scheduler.resize(2, &mut kv_cache);
    scheduler.slots[0].attach_request(request(1, "a"), admission(0));
    scheduler.slots[0].phase = SlotPhase::Prefill;
    scheduler.slots[0].prefill_cursor = 2;
    scheduler.slots[1].attach_request(request(2, "b"), admission(1));
    scheduler.slots[1].phase = SlotPhase::Prefill;
    scheduler.slots[1].prefill_cursor = 3;

    let mut ready = Vec::new();
    scheduler.select_prefill_ready_slots_into(&mut ready);
    assert_eq!(ready, vec![0]);
}

#[test]
fn admit_pending_request_leases_sequence_and_marks_manager_in_flight() {
    let mut scheduler = SlotScheduler::default();
    let mut kv_cache = KvCacheManager::new(1);
    scheduler.resize(1, &mut kv_cache);
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1, "ctx")));

    assert!(admit_one(&mut scheduler, &mut queue, &mut kv_cache).is_some());

    let slot = &scheduler.slots[0];
    assert_eq!(slot.request_id, 1);
    assert_eq!(slot.seq_id, 0);
    assert_eq!(slot.phase, SlotPhase::Prefill);
    assert!(!kv_cache.can_admit("other"));
}

#[test]
fn finalize_completed_slot_writes_response_and_keeps_live_residency() {
    let mut scheduler = SlotScheduler::default();
    let mut kv_cache = KvCacheManager::new(1);
    scheduler.resize(1, &mut kv_cache);
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1, "ctx")));
    assert!(admit_one(&mut scheduler, &mut queue, &mut kv_cache).is_some());

    let slot = &mut scheduler.slots[0];
    slot.phase = SlotPhase::Completed;
    slot.output_text = "done".to_string();
    slot.mirror.current_kv_tokens = vec![1, 2, 3, 4];
    slot.mirror.n_past = 4;

    scheduler.finalize_completed_slots(&mut queue, &mut kv_cache, KvReuseMode::LiveSlotPrefix);

    let response = queue.completed_responses.get(&1).expect("response");
    assert_eq!(response.status, GenerateResponseStatus::Completed);
    assert_eq!(response.output, ResponseOutput::Text("done".to_string()));
    assert_eq!(scheduler.slots[0].phase, SlotPhase::Idle);

    let warm = kv_cache
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("warm admission");
    assert_eq!(warm.candidate, CacheCandidate::Live);
    assert_eq!(warm.mirror.current_kv_tokens, vec![1, 2, 3, 4]);
}

#[test]
fn finalize_failed_slot_writes_terminal_error() {
    let mut scheduler = SlotScheduler::default();
    let mut kv_cache = KvCacheManager::new(1);
    scheduler.resize(1, &mut kv_cache);
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1, "ctx")));
    assert!(admit_one(&mut scheduler, &mut queue, &mut kv_cache).is_some());

    let slot = &mut scheduler.slots[0];
    slot.phase = SlotPhase::Failed;
    slot.terminal_error_message = "decode failed".to_string();

    scheduler.finalize_completed_slots(&mut queue, &mut kv_cache, KvReuseMode::LiveSlotPrefix);

    let response = queue.completed_responses.get(&1).expect("response");
    assert_eq!(response.status, GenerateResponseStatus::Failed);
    assert_eq!(response.error_message, "decode failed");
}

#[test]
fn finalize_cancelled_slot_prefers_cancel_message() {
    let mut scheduler = SlotScheduler::default();
    let mut kv_cache = KvCacheManager::new(1);
    scheduler.resize(1, &mut kv_cache);
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1, "ctx")));
    assert!(admit_one(&mut scheduler, &mut queue, &mut kv_cache).is_some());
    assert!(queue.cancel(1, "cancelled by caller".to_string()));

    let slot = &mut scheduler.slots[0];
    slot.phase = SlotPhase::Failed;
    slot.terminal_error_message = "decode failed".to_string();

    scheduler.finalize_completed_slots(&mut queue, &mut kv_cache, KvReuseMode::LiveSlotPrefix);

    let response = queue.completed_responses.get(&1).expect("response");
    assert_eq!(response.status, GenerateResponseStatus::Cancelled);
    assert_eq!(
        response.error_message,
        crate::runtime::REQUEST_CANCELLED_MESSAGE
    );
}

#[test]
fn emit_buffered_piece_appends_output_and_stream_frame_when_enabled() {
    let mut queue = RequestQueue::new();
    let (producer, consumer) = token_byte_ring(1024);
    queue.token_emission_sinks.insert(1, Arc::new(producer));
    let mut slot = SlotState::new(0);
    let mut request = request(1, "ctx");
    request.emit_tokens = true;
    slot.attach_request(request, admission(0));
    slot.buffered_output_text = "tok".to_string();

    SlotScheduler::emit_buffered_token_piece(&mut queue, &mut slot);

    assert_eq!(slot.output_text, "tok");
    assert_eq!(queue.total_emitted_token_count, 1);
    assert!(queue.flush_token_emissions());
    let drain = consumer.drain_available(16, 1024);
    assert_eq!(drain.frames.len(), 1);
    assert_eq!(drain.frames[0].stream_id, 1);
    assert_eq!(drain.frames[0].bytes, b"tok");
}
