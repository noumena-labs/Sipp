//! Lock-protected byte ring for handing emitted token text from the engine thread to consumer threads with bounded buffering.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::sync::MutexGuard;
use std::time::Duration;

pub const TOKEN_RING_DEFAULT_CAPACITY: usize = 256 * 1024;
pub const TOKEN_RING_RECORD_HEADER_BYTES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenRingFrame {
    pub stream_id: u32,
    pub sequence: u32,
    pub flags: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenRingDrain {
    pub frames: Vec<TokenRingFrame>,
    pub drop_count: u64,
    pub closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenRingDrainStatus {
    pub frames_drained: usize,
    pub bytes_drained: usize,
    pub drop_count: u64,
    pub closed: bool,
}

#[derive(Debug)]
struct TokenByteRingInner {
    state: Mutex<TokenByteRingState>,
    available: Condvar,
    drop_count: AtomicU64,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenRingRecordHeader {
    stream_id: u32,
    sequence: u32,
    flags: u32,
    byte_len: u32,
}

impl TokenRingRecordHeader {
    fn encode(self) -> [u8; TOKEN_RING_RECORD_HEADER_BYTES] {
        let mut bytes = [0_u8; TOKEN_RING_RECORD_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&self.stream_id.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.flags.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.byte_len.to_le_bytes());
        bytes
    }

    fn decode(bytes: [u8; TOKEN_RING_RECORD_HEADER_BYTES]) -> Self {
        Self {
            stream_id: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            sequence: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            flags: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            byte_len: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        }
    }
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
        drop_count: AtomicU64::new(0),
    });
    (
        TokenByteRingProducer {
            inner: Arc::clone(&inner),
        },
        TokenByteRingConsumer { inner },
    )
}

impl TokenByteRingProducer {
    pub fn try_write_frame(&self, stream_id: u32, flags: u32, bytes: &[u8]) -> bool {
        if stream_id == 0 || bytes.is_empty() {
            return true;
        }

        let mut state = lock_ring_state(&self.inner.state);
        if state.closed {
            return false;
        }

        let Ok(byte_len) = u32::try_from(bytes.len()) else {
            self.inner.drop_count.fetch_add(1, Ordering::Relaxed);
            self.inner.available.notify_one();
            return false;
        };

        let Some(record_size) = TOKEN_RING_RECORD_HEADER_BYTES.checked_add(bytes.len()) else {
            self.inner.drop_count.fetch_add(1, Ordering::Relaxed);
            self.inner.available.notify_one();
            return false;
        };
        let Some(next_used) = state.used.checked_add(record_size) else {
            self.inner.drop_count.fetch_add(1, Ordering::Relaxed);
            self.inner.available.notify_one();
            return false;
        };
        if record_size > state.buffer.len() || next_used > state.buffer.len() {
            self.inner.drop_count.fetch_add(1, Ordering::Relaxed);
            self.inner.available.notify_one();
            return false;
        }

        let was_empty = state.used == 0;
        let record_sequence = next_sequence_for_stream(&mut state, stream_id);

        let offset = state.write_index;
        let header = TokenRingRecordHeader {
            stream_id,
            sequence: record_sequence,
            flags,
            byte_len,
        }
        .encode();
        write_bytes(&mut state.buffer, offset, &header);
        write_bytes(
            &mut state.buffer,
            offset + TOKEN_RING_RECORD_HEADER_BYTES,
            bytes,
        );
        state.write_index = (state.write_index + record_size) % state.buffer.len();
        state.used = next_used;
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
        if state.used > 0 || state.closed {
            return true;
        }
        let (state, _timeout) = match self.inner.available.wait_timeout(state, timeout) {
            Ok(result) => result,
            Err(error) => error.into_inner(),
        };
        state.used > 0 || state.closed
    }

    pub fn drain_available(&self, max_frames: usize, max_bytes: usize) -> TokenRingDrain {
        let mut frames = Vec::with_capacity(max_frames);
        let status = self.drain_into(&mut frames, max_frames, max_bytes);
        TokenRingDrain {
            frames,
            drop_count: status.drop_count,
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
        let possible_frame_count = state
            .used
            .checked_div(TOKEN_RING_RECORD_HEADER_BYTES)
            .unwrap_or(0)
            .min(max_frames);
        frames.reserve(possible_frame_count);
        let mut drained_frames = 0usize;
        let mut drained_bytes = 0usize;

        while state.used >= TOKEN_RING_RECORD_HEADER_BYTES && drained_frames < max_frames {
            let offset = state.read_index;
            let header = TokenRingRecordHeader::decode(read_record_header(&state.buffer, offset));
            let Ok(byte_len) = usize::try_from(header.byte_len) else {
                break;
            };
            let Some(record_size) = TOKEN_RING_RECORD_HEADER_BYTES.checked_add(byte_len) else {
                break;
            };
            if record_size > state.used {
                break;
            }
            let Some(next_drained_bytes) = drained_bytes.checked_add(byte_len) else {
                break;
            };
            if drained_bytes > 0 && next_drained_bytes > max_bytes {
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
                flags: header.flags,
                bytes,
            });
        }

        TokenRingDrainStatus {
            frames_drained: drained_frames,
            bytes_drained: drained_bytes,
            drop_count: self.inner.drop_count.load(Ordering::Relaxed),
            closed: state.closed && state.used == 0,
        }
    }
}

fn lock_ring_state(state: &Mutex<TokenByteRingState>) -> MutexGuard<'_, TokenByteRingState> {
    match state.lock() {
        Ok(state) => state,
        Err(error) => error.into_inner(),
    }
}

fn next_sequence_for_stream(state: &mut TokenByteRingState, stream_id: u32) -> u32 {
    if state.cached_stream_id == stream_id {
        let sequence = state.cached_next_sequence;
        state.cached_next_sequence = sequence.wrapping_add(1);
        return sequence;
    }

    if state.cached_stream_id != 0 {
        state
            .next_sequences
            .insert(state.cached_stream_id, state.cached_next_sequence);
    }
    let sequence = state.next_sequences.get(&stream_id).copied().unwrap_or(0);
    state.cached_stream_id = stream_id;
    state.cached_next_sequence = sequence.wrapping_add(1);
    sequence
}

fn write_bytes(buffer: &mut [u8], offset: usize, bytes: &[u8]) {
    let len = buffer.len();
    let offset = offset % len;
    let tail = len - offset;
    if bytes.len() <= tail {
        buffer[offset..offset + bytes.len()].copy_from_slice(bytes);
    } else {
        buffer[offset..].copy_from_slice(&bytes[..tail]);
        buffer[..bytes.len() - tail].copy_from_slice(&bytes[tail..]);
    }
}

fn read_record_header(buffer: &[u8], offset: usize) -> [u8; TOKEN_RING_RECORD_HEADER_BYTES] {
    let capacity = buffer.len();
    let offset = offset % capacity;
    let tail = capacity - offset;
    let mut header = [0_u8; TOKEN_RING_RECORD_HEADER_BYTES];
    if TOKEN_RING_RECORD_HEADER_BYTES <= tail {
        header.copy_from_slice(&buffer[offset..offset + TOKEN_RING_RECORD_HEADER_BYTES]);
    } else {
        header[..tail].copy_from_slice(&buffer[offset..]);
        header[tail..].copy_from_slice(&buffer[..TOKEN_RING_RECORD_HEADER_BYTES - tail]);
    }
    header
}

fn read_bytes(buffer: &[u8], offset: usize, len: usize) -> Vec<u8> {
    let capacity = buffer.len();
    let offset = offset % capacity;
    let tail = capacity - offset;
    if len <= tail {
        buffer[offset..offset + len].to_vec()
    } else {
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(&buffer[offset..]);
        out.extend_from_slice(&buffer[..len - tail]);
        out
    }
}

#[cfg(test)]
mod tests;
