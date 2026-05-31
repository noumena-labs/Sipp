//! Lock-protected byte ring for handing emitted token text from the engine thread to consumer threads.

use std::collections::HashMap;
use std::sync::MutexGuard;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::defaults::BYTES_PER_KIB;

pub const TOKEN_RING_DEFAULT_CAPACITY: usize = 256 * BYTES_PER_KIB;
pub const TOKEN_RING_RECORD_HEADER_BYTES: usize = 16;

mod record_io;

use record_io::{read_bytes, read_record_header, write_bytes, TokenRingRecordHeader};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenRingFrame {
    pub stream_id: u32,
    pub sequence: u32,
    pub frame_count: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenRingDrain {
    pub frames: Vec<TokenRingFrame>,
    pub closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenRingDrainStatus {
    pub frames_drained: usize,
    pub bytes_drained: usize,
    pub closed: bool,
}

#[derive(Debug)]
struct TokenByteRingInner {
    state: Mutex<TokenByteRingState>,
    available: Condvar,
}

#[derive(Debug)]
struct TokenByteRingState {
    buffer: Vec<u8>,
    read_index: usize,
    write_index: usize,
    used: usize,
    closed: bool,
    cached_stream_id: u32,
    cached_next_sequence: u32,
    next_sequences: HashMap<u32, u32>,
}

#[derive(Debug, Clone)]
pub struct TokenByteRingProducer {
    inner: Arc<TokenByteRingInner>,
}

#[derive(Debug, Clone)]
pub struct TokenByteRingConsumer {
    inner: Arc<TokenByteRingInner>,
}

pub fn token_byte_ring(capacity: usize) -> (TokenByteRingProducer, TokenByteRingConsumer) {
    let capacity = capacity.max(TOKEN_RING_RECORD_HEADER_BYTES);
    let inner = Arc::new(TokenByteRingInner {
        state: Mutex::new(TokenByteRingState {
            buffer: vec![0; capacity],
            read_index: 0,
            write_index: 0,
            used: 0,
            closed: false,
            cached_stream_id: 0,
            cached_next_sequence: 0,
            next_sequences: HashMap::new(),
        }),
        available: Condvar::new(),
    });
    (
        TokenByteRingProducer {
            inner: Arc::clone(&inner),
        },
        TokenByteRingConsumer { inner },
    )
}

impl TokenByteRingProducer {
    pub fn try_write_frame(&self, stream_id: u32, bytes: &[u8]) -> bool {
        self.try_write_batch(stream_id, 1, bytes)
    }

    pub fn try_write_batch(&self, stream_id: u32, frame_count: u32, bytes: &[u8]) -> bool {
        if frame_is_noop(stream_id, bytes) {
            return true;
        }

        let mut state = lock_ring_state(&self.inner.state);
        if state.closed {
            return false;
        }

        let Some(record) = writable_record(&mut state, bytes.len()) else {
            return false;
        };

        let was_empty = state.used == 0;
        let record_sequence = next_sequence_for_stream(&mut state, stream_id, frame_count);

        let offset = state.write_index;
        let header = TokenRingRecordHeader {
            stream_id,
            sequence: record_sequence,
            frame_count,
            byte_len: record.byte_len,
        }
        .encode();
        write_bytes(&mut state.buffer, offset, &header);
        write_bytes(
            &mut state.buffer,
            offset + TOKEN_RING_RECORD_HEADER_BYTES,
            bytes,
        );
        state.write_index = (state.write_index + record.size) % state.buffer.len();
        state.used = record.next_used;
        drop(state);
        if was_empty {
            self.inner.available.notify_one();
        }
        true
    }

    pub fn close(&self) {
        let mut state = lock_ring_state(&self.inner.state);
        state.closed = true;
        drop(state);
        self.inner.available.notify_all();
    }
}

impl TokenByteRingConsumer {
    pub fn wait_for_data(&self, timeout: Duration) -> bool {
        let state = lock_ring_state(&self.inner.state);
        if state.has_data_or_closed() {
            return true;
        }
        let (state, _timeout) = match self.inner.available.wait_timeout(state, timeout) {
            Ok(result) => result,
            Err(error) => error.into_inner(),
        };
        state.has_data_or_closed()
    }

    pub fn drain_available(&self, max_frames: usize, max_bytes: usize) -> TokenRingDrain {
        let mut frames = Vec::with_capacity(max_frames);
        let status = self.drain_into(&mut frames, max_frames, max_bytes);
        TokenRingDrain {
            frames,
            closed: status.closed,
        }
    }

    pub fn drain_into(
        &self,
        frames: &mut Vec<TokenRingFrame>,
        max_frames: usize,
        max_bytes: usize,
    ) -> TokenRingDrainStatus {
        let mut state = lock_ring_state(&self.inner.state);
        frames.reserve(possible_drain_frame_count(state.used, max_frames));
        let mut drained_frames = 0usize;
        let mut drained_bytes = 0usize;

        while state.used >= TOKEN_RING_RECORD_HEADER_BYTES && drained_frames < max_frames {
            let offset = state.read_index;
            let header = TokenRingRecordHeader::decode(read_record_header(&state.buffer, offset));
            let Ok(byte_len) = usize::try_from(header.byte_len) else {
                break;
            };
            let Some(record_size) = token_ring_record_size(byte_len) else {
                break;
            };
            if record_size > state.used {
                break;
            }
            let Some(next_drained_bytes) = drained_bytes.checked_add(byte_len) else {
                break;
            };
            if exceeds_followup_byte_limit(drained_bytes, next_drained_bytes, max_bytes) {
                break;
            }
            let bytes = read_bytes(
                &state.buffer,
                offset + TOKEN_RING_RECORD_HEADER_BYTES,
                byte_len,
            );
            state.read_index = (state.read_index + record_size) % state.buffer.len();
            state.used -= record_size;
            drained_bytes = next_drained_bytes;
            let Some(next_drained_frames) = drained_frames.checked_add(1) else {
                break;
            };
            drained_frames = next_drained_frames;
            frames.push(TokenRingFrame {
                stream_id: header.stream_id,
                sequence: header.sequence,
                frame_count: header.frame_count,
                bytes,
            });
        }

        TokenRingDrainStatus {
            frames_drained: drained_frames,
            bytes_drained: drained_bytes,
            closed: state.closed && state.used == 0,
        }
    }
}

impl TokenByteRingState {
    fn has_data_or_closed(&self) -> bool {
        self.used > 0 || self.closed
    }
}

fn frame_is_noop(stream_id: u32, bytes: &[u8]) -> bool {
    stream_id == 0 || bytes.is_empty()
}

fn lock_ring_state(state: &Mutex<TokenByteRingState>) -> MutexGuard<'_, TokenByteRingState> {
    match state.lock() {
        Ok(state) => state,
        Err(error) => error.into_inner(),
    }
}

fn token_ring_record_size(byte_len: usize) -> Option<usize> {
    TOKEN_RING_RECORD_HEADER_BYTES.checked_add(byte_len)
}

fn possible_drain_frame_count(used_bytes: usize, max_frames: usize) -> usize {
    used_bytes
        .checked_div(TOKEN_RING_RECORD_HEADER_BYTES)
        .unwrap_or(0)
        .min(max_frames)
}

fn exceeds_followup_byte_limit(
    drained_bytes: usize,
    next_drained_bytes: usize,
    max_bytes: usize,
) -> bool {
    drained_bytes > 0 && next_drained_bytes > max_bytes
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WritableRecord {
    byte_len: u32,
    size: usize,
    next_used: usize,
}

fn writable_record(state: &mut TokenByteRingState, byte_len: usize) -> Option<WritableRecord> {
    let byte_len_u32 = u32::try_from(byte_len).ok()?;
    let size = token_ring_record_size(byte_len)?;
    let next_used = state.used.checked_add(size)?;
    if (size > state.buffer.len() || next_used > state.buffer.len())
        && !grow_ring_buffer(state, size.max(next_used))
    {
        return None;
    }
    Some(WritableRecord {
        byte_len: byte_len_u32,
        size,
        next_used,
    })
}

fn grow_ring_buffer(state: &mut TokenByteRingState, min_capacity: usize) -> bool {
    let mut next_capacity = state.buffer.len().max(TOKEN_RING_RECORD_HEADER_BYTES);
    while next_capacity < min_capacity {
        let Some(grown) = next_capacity.checked_mul(2) else {
            return false;
        };
        next_capacity = grown;
    }

    let used = state.used;
    let mut next_buffer = vec![0; next_capacity];
    if used > 0 {
        let bytes = read_bytes(&state.buffer, state.read_index, used);
        next_buffer[..used].copy_from_slice(&bytes);
    }
    state.buffer = next_buffer;
    state.read_index = 0;
    state.write_index = used;
    true
}

fn next_sequence_for_stream(
    state: &mut TokenByteRingState,
    stream_id: u32,
    frame_count: u32,
) -> u32 {
    if state.cached_stream_id == stream_id {
        let sequence = state.cached_next_sequence;
        state.cached_next_sequence = sequence.wrapping_add(frame_count);
        return sequence;
    }

    if state.cached_stream_id != 0 {
        state
            .next_sequences
            .insert(state.cached_stream_id, state.cached_next_sequence);
    }
    let sequence = state.next_sequences.get(&stream_id).copied().unwrap_or(0);
    state.cached_stream_id = stream_id;
    state.cached_next_sequence = sequence.wrapping_add(frame_count);
    sequence
}

#[cfg(test)]
mod tests {
    mod token_ring_tests;
}
