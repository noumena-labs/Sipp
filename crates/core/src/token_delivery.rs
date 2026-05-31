/// Counters owned by the token data plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenDeliveryStats {
    pub frames_sent: u64,
    pub bytes_sent: u64,
    pub frames_dropped: u64,
    pub batches_sent: u64,
}

/// A batch of generated token text delivered through the token data plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBatch {
    pub request_id: String,
    pub stream_id: u32,
    pub sequence_start: u32,
    pub text: String,
    pub frame_count: u32,
    pub byte_count: u32,
    pub stats: TokenDeliveryStats,
}
