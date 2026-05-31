//! Unit tests for the parent module.

use super::super::*;

#[test]
fn drains_wrapped_records_in_order() {
    let (producer, consumer) = token_byte_ring(40);
    assert!(producer.try_write_frame(7, 0, b"abcdefghij"));
    let first = consumer.drain_available(16, 1024);
    assert_eq!(first.frames.len(), 1);
    assert_eq!(first.frames[0].bytes, b"abcdefghij");

    assert!(producer.try_write_frame(7, 0, b"klmnop"));
    assert!(producer.try_write_frame(7, 0, b"qr"));
    let second = consumer.drain_available(16, 1024);
    assert_eq!(second.frames.len(), 2);
    assert_eq!(second.frames[0].sequence, 1);
    assert_eq!(second.frames[0].bytes, b"klmnop");
    assert_eq!(second.frames[1].sequence, 2);
    assert_eq!(second.frames[1].bytes, b"qr");
}

#[test]
fn drain_into_reserves_outer_frame_capacity_from_ring_usage() {
    let (producer, consumer) = token_byte_ring(128);
    assert!(producer.try_write_frame(1, 0, b"a"));
    assert!(producer.try_write_frame(1, 0, b"b"));
    let mut frames = Vec::new();

    let status = consumer.drain_into(&mut frames, 8, 1024);

    assert_eq!(status.frames_drained, 2);
    assert_eq!(frames.len(), 2);
    assert!(frames.capacity() >= 2);
}

#[test]
fn drain_helpers_preserve_capacity_and_followup_byte_limit_rules() {
    assert_eq!(
        possible_drain_frame_count(TOKEN_RING_RECORD_HEADER_BYTES * 3, 2),
        2
    );
    assert_eq!(
        possible_drain_frame_count(TOKEN_RING_RECORD_HEADER_BYTES - 1, 8),
        0
    );
    assert!(!exceeds_followup_byte_limit(0, 9, 4));
    assert!(!exceeds_followup_byte_limit(2, 4, 4));
    assert!(exceeds_followup_byte_limit(2, 5, 4));
}

#[test]
fn drain_into_allows_first_frame_over_byte_limit() {
    let (producer, consumer) = token_byte_ring(128);
    assert!(producer.try_write_frame(1, 0, b"abcdef"));
    assert!(producer.try_write_frame(1, 0, b"gh"));

    let drained = consumer.drain_available(8, 4);

    assert_eq!(drained.frames.len(), 1);
    assert_eq!(drained.frames[0].bytes, b"abcdef");
}

#[test]
fn grows_when_full() {
    let (producer, consumer) = token_byte_ring(24);
    assert!(producer.try_write_frame(1, 0, b"abc"));
    assert!(producer.try_write_frame(1, 0, b"def"));

    let drained = consumer.drain_available(16, 1024);
    assert_eq!(drained.frames.len(), 2);
    assert_eq!(drained.frames[0].bytes, b"abc");
    assert_eq!(drained.frames[1].bytes, b"def");
}

#[test]
fn grows_for_frame_larger_than_ring_capacity() {
    let (producer, consumer) = token_byte_ring(24);

    assert!(producer.try_write_frame(1, 0, b"abcdefghi"));

    let drained = consumer.drain_available(16, 1024);
    assert_eq!(drained.frames.len(), 1);
    assert_eq!(drained.frames[0].bytes, b"abcdefghi");
}

#[test]
fn consumer_recovers_after_ring_mutex_poison() {
    let (producer, consumer) = token_byte_ring(128);
    let poisoner = producer.clone();
    let _ = std::thread::spawn(move || {
        let _guard = poisoner.inner.state.lock().expect("lock");
        panic!("poison token ring mutex");
    })
    .join();

    assert!(producer.try_write_frame(1, 0, b"ok"));
    assert!(consumer.wait_for_data(Duration::from_millis(0)));
    let drained = consumer.drain_available(16, 1024);

    assert_eq!(drained.frames.len(), 1);
    assert_eq!(drained.frames[0].bytes, b"ok");
}
