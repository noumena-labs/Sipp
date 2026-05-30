//! Driver-owned token stream batching over the runtime token byte ring.

use futures_channel::mpsc;

use crate::engine::stream::{StreamStats, TokenBatch};
use crate::runtime::numeric::saturating_usize_to_u32;
use crate::runtime::request::{
    token_byte_ring, TokenByteRingConsumer, TokenByteRingProducer, TokenRingFrame,
    TOKEN_RING_DEFAULT_CAPACITY,
};

use super::{TOKEN_BATCH_MAX_BYTES, TOKEN_BATCH_MAX_FRAMES};

pub(super) const TOKEN_STREAM_CHANNEL_CAPACITY: usize = 256;

pub(super) struct ActiveTokenStream {
    pub(super) producer: TokenByteRingProducer,
    consumer: TokenByteRingConsumer,
    state: TokenStreamState,
    batch_tx: mpsc::Sender<TokenBatch>,
    pending_dropped_frames: u64,
    frames: Vec<TokenRingFrame>,
}

pub(super) fn start_engine_token_stream(
    request_id: u32,
    batch_tx: mpsc::Sender<TokenBatch>,
) -> ActiveTokenStream {
    let (producer, consumer) = token_byte_ring(TOKEN_RING_DEFAULT_CAPACITY);
    ActiveTokenStream {
        producer,
        consumer,
        state: TokenStreamState::new(request_id),
        batch_tx,
        pending_dropped_frames: 0,
        frames: Vec::with_capacity(TOKEN_BATCH_MAX_FRAMES),
    }
}

pub(super) fn drain_ring_into_sender(token: &mut ActiveTokenStream) {
    loop {
        token.frames.clear();
        let mut latest_drop_count = token.state.last_drop_count;
        let mut byte_count = 0usize;
        let mut closed = false;

        loop {
            let remaining_frames = remaining_quota(TOKEN_BATCH_MAX_FRAMES, token.frames.len());
            let remaining_bytes = remaining_quota(TOKEN_BATCH_MAX_BYTES, byte_count);
            let drain =
                token
                    .consumer
                    .drain_into(&mut token.frames, remaining_frames, remaining_bytes);
            latest_drop_count = latest_drop_count.max(drain.drop_count);
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

        let Some(mut batch) = token_batch_from_ring_frames(
            &token.frames,
            token.state.request_id,
            &mut token.state,
            latest_drop_count,
        ) else {
            break;
        };

        if token.pending_dropped_frames > 0 {
            batch.stats.frames_dropped = batch
                .stats
                .frames_dropped
                .saturating_add(token.pending_dropped_frames);
            token.pending_dropped_frames = 0;
        }

        if let Err(error) = token.batch_tx.try_send(batch) {
            if error.is_full() {
                token.pending_dropped_frames = token
                    .pending_dropped_frames
                    .saturating_add(u64::from(error.into_inner().frame_count));
            }
            break;
        }
    }
}

pub(super) struct TokenStreamState {
    request_id: u32,
    next_sequence: u32,
    last_drop_count: u64,
    stats: StreamStats,
}

impl TokenStreamState {
    pub(super) fn new(request_id: u32) -> Self {
        Self {
            request_id,
            next_sequence: 0,
            last_drop_count: 0,
            stats: StreamStats::default(),
        }
    }
}

pub(super) fn token_batch_from_ring_frames(
    frames: &[TokenRingFrame],
    target_request_id: u32,
    token_state: &mut TokenStreamState,
    drop_count: u64,
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
        frame_count = frame_count.saturating_add(1);
        byte_count = byte_count.saturating_add(saturating_usize_to_u32(frame.bytes.len()));
    }

    update_stream_drop_stats(token_state, drop_count);

    if frame_count == 0 {
        return None;
    }

    token_state.next_sequence = sequence_start
        .unwrap_or(token_state.next_sequence)
        .saturating_add(frame_count);
    update_stream_sent_stats(&mut token_state.stats, frame_count, byte_count);

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

fn update_stream_drop_stats(token_state: &mut TokenStreamState, drop_count: u64) {
    let drop_delta = drop_count.saturating_sub(token_state.last_drop_count);
    token_state.last_drop_count = drop_count;
    token_state.stats.frames_dropped = token_state.stats.frames_dropped.saturating_add(drop_delta);
}

fn update_stream_sent_stats(stats: &mut StreamStats, frame_count: u32, byte_count: u32) {
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
mod tests {
    mod token_stream_tests;
}
