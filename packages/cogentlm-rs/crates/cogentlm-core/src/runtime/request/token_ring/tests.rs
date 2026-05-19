//! Unit tests for the parent module.

use super::*;

#[test]
fn record_header_round_trips_without_slice_conversions() {
    let header = TokenRingRecordHeader {
        stream_id: 7,
        sequence: u32::MAX - 1,
        flags: 3,
        byte_len: 4096,
    };

    assert_eq!(TokenRingRecordHeader::decode(header.encode()), header);
}

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
fn counts_drops_when_full() {
    let (producer, consumer) = token_byte_ring(24);
    assert!(producer.try_write_frame(1, 0, b"abc"));
    assert!(!producer.try_write_frame(1, 0, b"def"));

    let drained = consumer.drain_available(16, 1024);
    assert_eq!(drained.frames.len(), 1);
    assert_eq!(drained.drop_count, 1);
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
