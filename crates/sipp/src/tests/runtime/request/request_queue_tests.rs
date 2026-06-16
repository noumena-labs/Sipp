//! Tests the `runtime::request::request_queue` module in `sipp`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use std::sync::Arc;

use super::*;
use crate::runtime::request::token_byte_ring;

fn request(id: GenerateRequestId) -> GenerateRequest {
    GenerateRequest::new(id, format!("ctx-{id}"))
}

#[test]
fn rejects_zero_and_duplicate_request_ids() {
    let mut queue = RequestQueue::new();
    assert!(!queue.push(request(0)));
    assert!(queue.push(request(1)));
    assert!(!queue.push(request(1)));
}

#[test]
fn pops_first_admissible_request_and_marks_admitted() {
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(1)));
    assert!(queue.push(request(2)));

    let popped = queue.try_pop_next_admissible(|request| request.id == 2);
    assert_eq!(popped, Some(2));
    assert_eq!(
        queue.request_lifecycle(2),
        Some(GenerateRequestLifecycle::Admitted)
    );
    assert_eq!(queue.try_pop_next_admissible(|_| true), Some(1));
}

#[test]
fn cancelling_pending_request_creates_completed_response() {
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(7)));
    assert!(queue.cancel(7, "cancelled".to_string()));
    assert_eq!(queue.try_pop_next_admissible(|_| true), None);

    let response = queue.completed_responses.get(&7).expect("response");
    assert_eq!(response.status, GenerateResponseStatus::Cancelled);
    assert_eq!(response.error_message, "cancelled");
}

#[test]
fn cancelling_admitted_request_marks_it_for_runtime_cancellation() {
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(8)));
    assert_eq!(queue.try_pop_next_admissible(|_| true), Some(8));

    assert!(queue.cancel(8, "cancelled".to_string()));

    assert!(queue.request_cancel_requested(8));
    assert_eq!(
        queue.request_lifecycle(8),
        Some(GenerateRequestLifecycle::Admitted)
    );
    assert!(!queue.completed_responses.contains_key(&8));
}

#[test]
fn append_token_piece_without_ring_is_a_noop() {
    let mut queue = RequestQueue::new();
    queue.append_token_piece(1, "a");

    assert_eq!(queue.total_emitted_token_count, 0);
}

#[test]
fn append_token_piece_writes_to_token_ring() {
    let mut queue = RequestQueue::new();
    let (producer, consumer) = token_byte_ring(1024);
    queue.token_emission_sinks.insert(9, Arc::new(producer));

    queue.append_token_piece(9, "tok");
    assert_eq!(consumer.drain_available(16, 1024).frames.len(), 0);

    assert!(queue.flush_token_emissions());

    let drain = consumer.drain_available(16, 1024);
    assert_eq!(drain.frames.len(), 1);
    assert_eq!(drain.frames[0].stream_id, 9);
    assert_eq!(drain.frames[0].frame_count, 1);
    assert_eq!(drain.frames[0].bytes, b"tok");
    assert_eq!(queue.total_emitted_token_count, 1);
}

#[test]
fn flush_token_emissions_batches_pieces_per_request() {
    let mut queue = RequestQueue::new();
    let (producer, consumer) = token_byte_ring(1024);
    queue.token_emission_sinks.insert(9, Arc::new(producer));

    queue.append_token_piece(9, "to");
    queue.append_token_piece(9, "k");
    assert!(queue.flush_token_emissions());

    let drain = consumer.drain_available(16, 1024);
    assert_eq!(drain.frames.len(), 1);
    assert_eq!(drain.frames[0].stream_id, 9);
    assert_eq!(drain.frames[0].sequence, 0);
    assert_eq!(drain.frames[0].frame_count, 2);
    assert_eq!(drain.frames[0].bytes, b"tok");
    assert_eq!(queue.total_emitted_token_count, 2);

    queue.append_token_piece(9, "!");
    assert!(queue.flush_token_emissions());

    let next = consumer.drain_available(16, 1024);
    assert_eq!(next.frames.len(), 1);
    assert_eq!(next.frames[0].sequence, 2);
    assert_eq!(next.frames[0].frame_count, 1);
    assert_eq!(next.frames[0].bytes, b"!");
}

#[test]
fn emitted_token_count_saturates_at_i32_max() {
    let mut queue = RequestQueue::new();
    let (producer, consumer) = token_byte_ring(1024);
    queue.token_emission_sinks.insert(9, Arc::new(producer));
    queue.total_emitted_token_count = i32::MAX;

    queue.append_token_piece(9, "tok");
    assert!(queue.flush_token_emissions());

    assert_eq!(queue.total_emitted_token_count, i32::MAX);
    assert_eq!(consumer.drain_available(16, 1024).frames.len(), 1);
}

#[test]
fn flush_token_emissions_reports_noop_when_no_tokens_are_pending() {
    let mut queue = RequestQueue::new();
    let (producer, _consumer) = token_byte_ring(1024);
    queue.token_emission_sinks.insert(9, Arc::new(producer));

    assert!(!queue.flush_token_emissions());
}
