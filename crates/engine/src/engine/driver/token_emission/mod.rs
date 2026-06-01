//! Driver-owned token emission batching over the runtime token byte ring.

use futures_channel::mpsc;

use crate::engine::token_emission::{TokenBatch, TokenEmissionStats};
use crate::runtime::numeric::saturating_usize_to_u32;
use crate::runtime::request::{
    token_byte_ring, TokenByteRingConsumer, TokenEmissionSinkRef, TokenRingFrame,
    TOKEN_RING_DEFAULT_CAPACITY,
};

use super::{TOKEN_BATCH_MAX_BYTES, TOKEN_BATCH_MAX_FRAMES};

pub(super) struct ActiveTokenEmission {
    pub(super) producer: TokenEmissionSinkRef,
    consumer: TokenByteRingConsumer,
    state: TokenEmissionState,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
    frames: Vec<TokenRingFrame>,
}

pub(super) fn start_engine_token_emission(
    request_id: u32,
    batch_tx: mpsc::UnboundedSender<TokenBatch>,
) -> ActiveTokenEmission {
    let (producer, consumer) = token_byte_ring(TOKEN_RING_DEFAULT_CAPACITY);
    ActiveTokenEmission {
        producer: std::sync::Arc::new(producer),
        consumer,
        state: TokenEmissionState::new(request_id),
        batch_tx,
        frames: Vec::with_capacity(TOKEN_BATCH_MAX_FRAMES),
    }
}

pub(super) fn drain_ring_into_sender(token: &mut ActiveTokenEmission) {
    loop {
        token.frames.clear();
        let mut byte_count = 0usize;
        let mut closed = false;

        loop {
            let remaining_frames = remaining_quota(TOKEN_BATCH_MAX_FRAMES, token.frames.len());
            let remaining_bytes = remaining_quota(TOKEN_BATCH_MAX_BYTES, byte_count);
            let drain =
                token
                    .consumer
                    .drain_into(&mut token.frames, remaining_frames, remaining_bytes);
            closed |= drain.closed;
            let Some(next_byte_count) = next_batch_byte_count(byte_count, drain.bytes_drained)
            else {
                break;
            };
            byte_count = next_byte_count;

            if drain.frames_drained == 0
                || closed
                || token.frames.len() >= TOKEN_BATCH_MAX_FRAMES
                || byte_count >= TOKEN_BATCH_MAX_BYTES
            {
                break;
            }
        }

        let Some(batch) =
            token_batch_from_ring_frames(&token.frames, token.state.request_id, &mut token.state)
        else {
            break;
        };

        if token.batch_tx.unbounded_send(batch).is_err() {
            break;
        }
    }
}

pub(super) struct TokenEmissionState {
    request_id: u32,
    next_sequence: u32,
    stats: TokenEmissionStats,
}

impl TokenEmissionState {
    pub(super) fn new(request_id: u32) -> Self {
        Self {
            request_id,
            next_sequence: 0,
            stats: TokenEmissionStats::default(),
        }
    }
}

pub(super) fn token_batch_from_ring_frames(
    frames: &[TokenRingFrame],
    target_request_id: u32,
    token_state: &mut TokenEmissionState,
) -> Option<TokenBatch> {
    let text_capacity = matching_token_frames(frames, target_request_id)
        .map(|frame| frame.bytes.len())
        .sum();
    let mut text = String::with_capacity(text_capacity);
    let mut frame_count = 0_u32;
    let mut byte_count = 0_u32;
    let mut sequence_start = None;

    for frame in matching_token_frames(frames, target_request_id) {
        if sequence_start.is_none() {
            sequence_start = Some(frame.sequence);
        }
        match std::str::from_utf8(&frame.bytes) {
            Ok(piece) => text.push_str(piece),
            Err(_) => text.push_str(&String::from_utf8_lossy(&frame.bytes)),
        }
        frame_count = frame_count.saturating_add(frame.frame_count);
        byte_count = byte_count.saturating_add(saturating_usize_to_u32(frame.bytes.len()));
    }

    if frame_count == 0 {
        return None;
    }

    token_state.next_sequence = sequence_start
        .unwrap_or(token_state.next_sequence)
        .saturating_add(frame_count);
    update_emission_sent_stats(&mut token_state.stats, frame_count, byte_count);

    Some(TokenBatch {
        request_id: token_state.request_id.to_string(),
        stream_id: token_state.request_id,
        sequence_start: sequence_start.unwrap_or_default(),
        text,
        frame_count,
        byte_count,
        stats: token_state.stats,
    })
}

fn matching_token_frames(
    frames: &[TokenRingFrame],
    stream_id: u32,
) -> impl Iterator<Item = &TokenRingFrame> {
    frames
        .iter()
        .filter(move |frame| frame.stream_id == stream_id)
}

fn update_emission_sent_stats(stats: &mut TokenEmissionStats, frame_count: u32, byte_count: u32) {
    stats.frames_sent = stats.frames_sent.saturating_add(u64::from(frame_count));
    stats.bytes_sent = stats.bytes_sent.saturating_add(u64::from(byte_count));
    stats.batches_sent = stats.batches_sent.saturating_add(1);
}

fn next_batch_byte_count(current: usize, drained: usize) -> Option<usize> {
    current.checked_add(drained)
}

fn remaining_quota(limit: usize, used: usize) -> usize {
    limit.saturating_sub(used).max(1)
}

#[cfg(test)]
#[path = "../../../tests/engine/driver/token_emission_tests.rs"]
mod token_emission_tests;
