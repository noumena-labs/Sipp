//! Unit tests for the parent module.

use super::super::*;
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
        queue.find(2).map(|request| request.lifecycle),
        Some(GenerateRequestLifecycle::Admitted)
    );
    assert_eq!(queue.try_pop_next(), Some(1));
}

#[test]
fn cancelling_pending_request_creates_completed_response() {
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(7)));
    assert!(queue.cancel(7, "cancelled".to_string()));
    assert_eq!(queue.try_pop_next(), None);

    let response = queue.peek_completed_response(7).expect("response");
    assert_eq!(response.status, GenerateResponseStatus::Cancelled);
    assert_eq!(response.error_message, "cancelled");
}

#[test]
fn cancelling_admitted_request_marks_it_for_runtime_cancellation() {
    let mut queue = RequestQueue::new();
    assert!(queue.push(request(8)));
    assert_eq!(queue.try_pop_next(), Some(8));

    assert!(queue.cancel(8, "cancelled".to_string()));

    let request = queue.find(8).expect("admitted request");
    assert!(request.cancel_requested);
    assert_eq!(request.lifecycle, GenerateRequestLifecycle::Admitted);
    assert!(queue.peek_completed_response(8).is_none());
}

#[test]
fn completed_response_ids_are_sorted_for_deterministic_polling() {
    let mut queue = RequestQueue::new();
    for id in [3, 1, 2] {
        queue.mark_completed(GenerateResponse {
            request_id: id,
            status: GenerateResponseStatus::Completed,
            ..GenerateResponse::default()
        });
    }

    assert_eq!(queue.completed_response_ids(), vec![1, 2, 3]);
}

#[test]
fn append_streaming_token_without_ring_is_a_noop() {
    let mut queue = RequestQueue::new();
    queue.append_streaming_token(1, "a");

    assert_eq!(queue.total_emitted_token_count(), 0);
}

#[test]
fn append_streaming_token_writes_to_token_ring() {
    let mut queue = RequestQueue::new();
    let (producer, consumer) = token_byte_ring(1024);
    queue.add_token_ring_producer(9, producer);

    queue.append_streaming_token(9, "tok");

    let drain = consumer.drain_available(16, 1024);
    assert_eq!(drain.frames.len(), 1);
    assert_eq!(drain.frames[0].stream_id, 9);
    assert_eq!(drain.frames[0].bytes, b"tok");
    assert_eq!(queue.total_emitted_token_count(), 1);
}

#[test]
fn emitted_token_count_saturates_at_i32_max() {
    let mut queue = RequestQueue::new();
    let (producer, consumer) = token_byte_ring(1024);
    queue.add_token_ring_producer(9, producer);
    queue.total_emitted_token_count = i32::MAX;

    queue.append_streaming_token(9, "tok");

    assert_eq!(queue.total_emitted_token_count(), i32::MAX);
    assert_eq!(consumer.drain_available(16, 1024).frames.len(), 1);
}
