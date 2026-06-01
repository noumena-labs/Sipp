/// Counters owned by the token data plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenEmissionStats {
    /// Token frames emitted to the consumer.
    pub frames_sent: u64,
    /// UTF-8 payload bytes emitted to the consumer.
    pub bytes_sent: u64,
    /// Token batches emitted to the consumer.
    pub batches_sent: u64,
}

/// A batch of generated token text emitted through the token data plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBatch {
    /// Stable request identifier for this batch.
    pub request_id: String,
    /// Numeric stream identifier used by local runtimes.
    pub stream_id: u32,
    /// Sequence index of the first token frame in this batch.
    pub sequence_start: u32,
    /// Concatenated token text for this batch.
    pub text: String,
    /// Number of token frames represented by this batch.
    pub frame_count: u32,
    /// UTF-8 payload bytes represented by this batch.
    pub byte_count: u32,
    /// Cumulative token-emission counters after this batch.
    pub stats: TokenEmissionStats,
}
