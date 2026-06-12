/// Why a request stopped producing tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

impl FinishReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        }
    }
}
