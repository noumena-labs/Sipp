use std::fmt;

/// Canonical inference operation used for capability and endpoint routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Raw-prompt text generation.
    Query,
    /// Message-shaped text generation.
    Chat,
    /// Vector embedding.
    Embed,
    /// Encoded-audio transcription.
    Listen,
    /// Text-to-WAV synthesis.
    Speak,
}

impl Operation {
    /// Return the stable public operation label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Chat => "chat",
            Self::Embed => "embed",
            Self::Listen => "listen",
            Self::Speak => "speak",
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}
