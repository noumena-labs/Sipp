use super::TOKEN_RING_RECORD_HEADER_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TokenRingRecordHeader {
    pub stream_id: u32,
    pub sequence: u32,
    pub flags: u32,
    pub byte_len: u32,
}

impl TokenRingRecordHeader {
    pub(super) fn encode(self) -> [u8; TOKEN_RING_RECORD_HEADER_BYTES] {
        let mut bytes = [0_u8; TOKEN_RING_RECORD_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&self.stream_id.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.flags.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.byte_len.to_le_bytes());
        bytes
    }

    pub(super) fn decode(bytes: [u8; TOKEN_RING_RECORD_HEADER_BYTES]) -> Self {
        Self {
            stream_id: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            sequence: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            flags: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
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

pub(super) fn read_bytes(buffer: &[u8], offset: usize, len: usize) -> Vec<u8> {
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
mod tests {
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
}
