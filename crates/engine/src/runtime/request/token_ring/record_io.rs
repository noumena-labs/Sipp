use super::TOKEN_RING_RECORD_HEADER_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TokenRingRecordHeader {
    pub stream_id: u32,
    pub sequence: u32,
    pub frame_count: u32,
    pub byte_len: u32,
}

impl TokenRingRecordHeader {
    pub(super) fn encode(self) -> [u8; TOKEN_RING_RECORD_HEADER_BYTES] {
        let mut bytes = [0_u8; TOKEN_RING_RECORD_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&self.stream_id.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.frame_count.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.byte_len.to_le_bytes());
        bytes
    }

    pub(super) fn decode(bytes: [u8; TOKEN_RING_RECORD_HEADER_BYTES]) -> Self {
        Self {
            stream_id: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            sequence: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            frame_count: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            byte_len: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        }
    }
}

pub(super) fn write_bytes(buffer: &mut [u8], offset: usize, bytes: &[u8]) {
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

pub(super) fn read_record_header(
    buffer: &[u8],
    offset: usize,
) -> [u8; TOKEN_RING_RECORD_HEADER_BYTES] {
    let mut header = [0_u8; TOKEN_RING_RECORD_HEADER_BYTES];
    read_bytes_into(buffer, offset, &mut header);
    header
}

pub(super) fn read_bytes(buffer: &[u8], offset: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    read_bytes_into(buffer, offset, &mut out);
    out
}

fn read_bytes_into(buffer: &[u8], offset: usize, out: &mut [u8]) {
    let out_len = out.len();
    let capacity = buffer.len();
    let offset = offset % capacity;
    let tail = capacity - offset;
    let first_len = out_len.min(tail);
    out[..first_len].copy_from_slice(&buffer[offset..offset + first_len]);
    if first_len < out_len {
        out[first_len..].copy_from_slice(&buffer[..out_len - first_len]);
    }
}

#[cfg(test)]
#[path = "../../../tests/runtime/request/token_ring/record_io_tests.rs"]
mod record_io_tests;
