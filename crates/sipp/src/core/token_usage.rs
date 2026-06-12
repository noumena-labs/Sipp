/// Provider-neutral token accounting shared by local and provider results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenUsage {
    /// Number of input tokens reported for the request.
    pub input_tokens: Option<u32>,
    /// Number of output tokens reported for the request.
    pub output_tokens: Option<u32>,
    /// Total token count reported for the request.
    pub total_tokens: Option<u32>,
}
