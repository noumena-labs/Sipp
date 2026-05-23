//! Background token sink: drains a [`TokenByteRingConsumer`] on its own thread
//! and invokes the caller's on-tokens callback with batched [`TokenBatch`]es.
//!
//! Batches are sized for either max frames, max bytes, or a short flush
//! interval — whichever lands first. Drop counts from the producer are
//! folded into the stream stats so consumers can see when backpressure hit.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::engine::stream::{StreamStats, TokenBatch};
use crate::error::{Error, Result};
use crate::runtime::numeric::saturating_usize_to_u32;
use crate::runtime::request::{
    token_byte_ring, TokenByteRingConsumer, TokenByteRingProducer, TokenRingFrame,
    TOKEN_RING_DEFAULT_CAPACITY,
};

use super::{
    runtime_command, OnTokensCallback, TOKEN_BATCH_FLUSH_INTERVAL, TOKEN_BATCH_MAX_BYTES,
    TOKEN_BATCH_MAX_FRAMES, TOKEN_CALLBACK_THREAD_PANICKED,
};

pub(super) struct AsyncTokenSink {
    pub(super) producer: TokenByteRingProducer,
    join_handle: Option<JoinHandle<()>>,
    error_rx: mpsc::Receiver<Error>,
}

impl AsyncTokenSink {
    pub(super) fn close(&self) {
        self.producer.close();
    }

    pub(super) fn try_recv_error(&mut self) -> Option<Error> {
        self.error_rx.try_recv().ok()
    }

    pub(super) fn join(&mut self) -> Result<()> {
        if let Some(join_handle) = self.join_handle.take() {
            join_handle
                .join()
                .map_err(|_| runtime_command(TOKEN_CALLBACK_THREAD_PANICKED))?;
        }
        if let Some(error) = self.try_recv_error() {
            return Err(error);
        }
        Ok(())
    }
}

pub(super) fn start_async_token_sink(
    request_id: u32,
    callback: OnTokensCallback,
) -> AsyncTokenSink {
    let (producer, consumer) = token_byte_ring(TOKEN_RING_DEFAULT_CAPACITY);
    let (error_tx, error_rx) = mpsc::channel();
    let join_handle = thread::spawn(move || {
        run_token_callback_loop(request_id, consumer, callback, error_tx);
    });
    AsyncTokenSink {
        producer,
        join_handle: Some(join_handle),
        error_rx,
    }
}

fn run_token_callback_loop(
    request_id: u32,
    consumer: TokenByteRingConsumer,
    mut callback: OnTokensCallback,
    error_tx: mpsc::Sender<Error>,
) {
    let mut token_state = TokenStreamState::new(request_id);
    let mut frames = Vec::with_capacity(TOKEN_BATCH_MAX_FRAMES);
    loop {
        consumer.wait_for_data(TOKEN_BATCH_FLUSH_INTERVAL);
        let batch_started = Instant::now();
        frames.clear();
        let mut latest_drop_count = token_state.last_drop_count;
        let mut closed = false;
        let mut byte_count = 0usize;

        loop {
            let remaining_frames = remaining_quota(TOKEN_BATCH_MAX_FRAMES, frames.len());
            let remaining_bytes = remaining_quota(TOKEN_BATCH_MAX_BYTES, byte_count);
            let drain = consumer.drain_into(&mut frames, remaining_frames, remaining_bytes);
            latest_drop_count = latest_drop_count.max(drain.drop_count);
            closed |= drain.closed;
            let Some(next_byte_count) = next_batch_byte_count(byte_count, drain.bytes_drained)
            else {
                break;
            };
            byte_count = next_byte_count;

            if closed
                || frames.len() >= TOKEN_BATCH_MAX_FRAMES
                || byte_count >= TOKEN_BATCH_MAX_BYTES
                || batch_started.elapsed() >= TOKEN_BATCH_FLUSH_INTERVAL
            {
                break;
            }

            let remaining = remaining_flush_interval(batch_started);
            if remaining.is_zero() || !consumer.wait_for_data(remaining) {
                break;
            }
        }

        if let Some(batch) =
            token_batch_from_ring_frames(&frames, request_id, &mut token_state, latest_drop_count)
        {
            if let Err(error) = callback(&batch) {
                let _ = error_tx.send(error);
                return;
            }
        }

        if closed {
            return;
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

fn remaining_flush_interval(batch_started: Instant) -> Duration {
    TOKEN_BATCH_FLUSH_INTERVAL
        .checked_sub(batch_started.elapsed())
        .unwrap_or(Duration::ZERO)
}

fn remaining_quota(limit: usize, used: usize) -> usize {
    limit.saturating_sub(used).max(1)
}

#[cfg(test)]
mod tests {
    mod token_sink_tests;
}
