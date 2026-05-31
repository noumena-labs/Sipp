use crate::runtime::request::{
    token_byte_ring, GenerateRequest, GenerateRequestId, GenerateResponseStatus, RequestQueue,
    ResponseOutput,
};
use crate::runtime::scheduler::SlotPhase;
use crate::runtime::session::{SequenceState, SessionStore};

use super::super::*;

fn request(id: GenerateRequestId, context_key: &str) -> GenerateRequest {
    let mut request = GenerateRequest::new(id, context_key);
    request.prompt_tokens = vec![1, 2, 3];
    request
}

#[test]
fn resize_resets_non_idle_slots_and_preserves_slot_ids() {
    let mut scheduler = SlotScheduler::default();
    scheduler.resize(2);
    scheduler.slots[0].attach_request(request(1, "a"), SequenceState::default());
    scheduler.slots[0].phase = SlotPhase::Decode;

    scheduler.resize(2);

    assert_eq!(scheduler.slots[0].slot_id, 0);
    assert_eq!(scheduler.slots[0].phase, SlotPhase::Idle);
    assert_eq!(scheduler.slots[1].slot_id, 1);
}

#[test]
fn selects_decode_ready_slots_without_buffered_text() {
    let mut scheduler = SlotScheduler::default();
    scheduler.resize(3);
    scheduler.slots[0].attach_request(request(1, "a"), SequenceState::default());
    scheduler.slots[0].session = Some(SequenceState::default());
    scheduler.slots[0].phase = SlotPhase::Decode;
    scheduler.slots[0].generated_tokens.push(10);
    scheduler.slots[1].attach_request(request(2, "b"), SequenceState::default());
    scheduler.slots[1].session = Some(SequenceState::default());
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
    scheduler.resize(2);
    scheduler.slots[0].attach_request(request(1, "a"), SequenceState::default());
    scheduler.slots[0].session = Some(SequenceState::default());
    scheduler.slots[0].phase = SlotPhase::Prefill;
    scheduler.slots[0].prefill_cursor = 2;
    scheduler.slots[1].attach_request(request(2, "b"), SequenceState::default());
    scheduler.slots[1].session = Some(SequenceState::default());
    scheduler.slots[1].phase = SlotPhase::Prefill;
    scheduler.slots[1].prefill_cursor = 3;

    let mut ready = Vec::new();
    scheduler.select_prefill_ready_slots_into(&mut ready);
    assert_eq!(ready, vec![0]);
}

#[test]
fn admit_pending_request_leases_sequence_and_pins_session() {
    let mut scheduler = SlotScheduler::default();
    scheduler.resize(1);
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1, "ctx")));
    let mut sessions = SessionStore::new(2, 1);

    assert!(scheduler
        .admit_pending_requests(&mut queue, &mut sessions)
        .is_some());

    let slot = &scheduler.slots[0];
    assert_eq!(slot.request_id, 1);
    assert_eq!(slot.seq_id, 0);
    assert_eq!(slot.phase, SlotPhase::Prefill);
    assert_eq!(
        sessions.find("ctx").map(|session| session.pin_count),
        Some(1)
    );
    assert!(!sessions.can_admit("other"));
}

#[test]
fn finalize_completed_slot_writes_response_and_releases_session() {
    let mut scheduler = SlotScheduler::default();
    scheduler.resize(1);
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1, "ctx")));
    let mut sessions = SessionStore::new(2, 1);
    assert!(scheduler
        .admit_pending_requests(&mut queue, &mut sessions)
        .is_some());

    let slot = &mut scheduler.slots[0];
    slot.phase = SlotPhase::Completed;
    slot.output_text = "done".to_string();
    slot.mirror.current_kv_tokens = vec![1, 2, 3, 4];
    slot.mirror.n_past = 4;

    scheduler.finalize_completed_slots(&mut queue, &mut sessions);

    let response = queue.completed_responses.get(&1).expect("response");
    assert_eq!(response.status, GenerateResponseStatus::Completed);
    assert_eq!(response.output, ResponseOutput::Text("done".to_string()));
    assert_eq!(
        sessions.find("ctx").map(|session| session.pin_count),
        Some(0)
    );
    assert_eq!(
        sessions
            .find("ctx")
            .map(|session| session.current_kv_tokens.clone()),
        Some(vec![1, 2, 3, 4])
    );
    assert_eq!(scheduler.slots[0].phase, SlotPhase::Idle);
    assert!(sessions.can_admit("other"));
}

#[test]
fn finalize_failed_slot_writes_terminal_error() {
    let mut scheduler = SlotScheduler::default();
    scheduler.resize(1);
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1, "ctx")));
    let mut sessions = SessionStore::new(2, 1);
    assert!(scheduler
        .admit_pending_requests(&mut queue, &mut sessions)
        .is_some());

    let slot = &mut scheduler.slots[0];
    slot.phase = SlotPhase::Failed;
    slot.terminal_error_message = "decode failed".to_string();

    scheduler.finalize_completed_slots(&mut queue, &mut sessions);

    let response = queue.completed_responses.get(&1).expect("response");
    assert_eq!(response.status, GenerateResponseStatus::Failed);
    assert_eq!(response.error_message, "decode failed");
}

#[test]
fn finalize_cancelled_slot_prefers_cancel_message() {
    let mut scheduler = SlotScheduler::default();
    scheduler.resize(1);
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1, "ctx")));
    let mut sessions = SessionStore::new(2, 1);
    assert!(scheduler
        .admit_pending_requests(&mut queue, &mut sessions)
        .is_some());
    assert!(queue.cancel(1, "cancelled by caller".to_string()));

    let slot = &mut scheduler.slots[0];
    slot.phase = SlotPhase::Failed;
    slot.terminal_error_message = "decode failed".to_string();

    scheduler.finalize_completed_slots(&mut queue, &mut sessions);

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
    queue.token_ring_producers.insert(1, producer);
    let mut slot = SlotState::new(0);
    let mut request = request(1, "ctx");
    request.emit_tokens = true;
    slot.attach_request(request, SequenceState::default());
    slot.buffered_output_text = "tok".to_string();

    SlotScheduler::emit_buffered_token_piece(&mut queue, &mut slot);

    assert_eq!(slot.output_text, "tok");
    assert_eq!(queue.total_emitted_token_count, 1);
    let drain = consumer.drain_available(16, 1024);
    assert_eq!(drain.frames.len(), 1);
    assert_eq!(drain.frames[0].stream_id, 1);
    assert_eq!(drain.frames[0].bytes, b"tok");
}
