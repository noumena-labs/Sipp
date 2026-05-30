//! Unit tests for the token sink module.

use super::super::*;
use crate::runtime::request::TokenRingFrame;

#[test]
fn token_ring_frames_are_batched_by_request() {
    let frames = vec![
        TokenRingFrame {
            stream_id: 1,
            sequence: 0,
            flags: 0,
            bytes: b"hel".to_vec(),
        },
        TokenRingFrame {
            stream_id: 2,
            sequence: 0,
            flags: 0,
            bytes: b"skip".to_vec(),
        },
        TokenRingFrame {
            stream_id: 1,
            sequence: 1,
            flags: 0,
            bytes: b"lo".to_vec(),
        },
    ];
    let mut state = TokenStreamState::new(1);

    let batch = token_batch_from_ring_frames(&frames, 1, &mut state, 0).expect("token batch");

    assert_eq!(batch.request_id, "1");
    assert_eq!(batch.stream_id, 1);
    assert_eq!(batch.sequence_start, 0);
    assert_eq!(batch.text, "hello");
    assert_eq!(batch.frame_count, 2);
    assert_eq!(batch.byte_count, 5);
    assert_eq!(batch.stats.frames_sent, 2);
    assert_eq!(batch.stats.bytes_sent, 5);
    assert_eq!(batch.stats.batches_sent, 1);
}

#[test]
fn token_ring_batch_tracks_drops_and_sequences() {
    let first = [TokenRingFrame {
        stream_id: 3,
        sequence: 0,
        flags: 0,
        bytes: b"a".to_vec(),
    }];
    let second = [TokenRingFrame {
        stream_id: 3,
        sequence: 1,
        flags: 0,
        bytes: b"bc".to_vec(),
    }];
    let mut state = TokenStreamState::new(3);

    let first = token_batch_from_ring_frames(&first, 3, &mut state, 2).expect("first batch");
    let second = token_batch_from_ring_frames(&second, 3, &mut state, 5).expect("second batch");

    assert_eq!(first.sequence_start, 0);
    assert_eq!(first.stats.frames_dropped, 2);
    assert_eq!(second.sequence_start, 1);
    assert_eq!(second.stats.frames_sent, 2);
    assert_eq!(second.stats.frames_dropped, 5);
    assert_eq!(second.stats.bytes_sent, 3);
    assert_eq!(second.stats.batches_sent, 2);
}

#[test]
fn token_ring_batch_stats_saturate() {
    let frames = [TokenRingFrame {
        stream_id: 1,
        sequence: 0,
        flags: 0,
        bytes: b"a".to_vec(),
    }];
    let mut state = TokenStreamState::new(1);
    state.stats.frames_sent = u64::MAX;
    state.stats.bytes_sent = u64::MAX;
    state.stats.batches_sent = u64::MAX;

    let batch = token_batch_from_ring_frames(&frames, 1, &mut state, 0).expect("token batch");

    assert_eq!(batch.stats.frames_sent, u64::MAX);
    assert_eq!(batch.stats.bytes_sent, u64::MAX);
    assert_eq!(batch.stats.batches_sent, u64::MAX);
}

#[test]
fn token_batch_byte_count_saturates() {
    assert_eq!(saturating_usize_to_u32(u32::MAX as usize + 1), u32::MAX);
}

#[test]
fn token_batch_byte_count_rejects_overflow() {
    assert_eq!(next_batch_byte_count(7, 5), Some(12));
    assert_eq!(next_batch_byte_count(usize::MAX, 1), None);
}

#[test]
fn remaining_quota_keeps_drain_progress_after_limit() {
    assert_eq!(remaining_quota(10, 3), 7);
    assert_eq!(remaining_quota(10, 10), 1);
    assert_eq!(remaining_quota(10, 11), 1);
}
