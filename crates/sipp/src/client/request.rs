use crate::core::ChatMessage;
use crate::engine::SamplingRuntimeOverride;

use crate::client::EndpointRef;

/// Request-scoped metadata propagated through endpoint execution.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SippRequestContext {
    /// Canonical request identifier assigned by the application boundary.
    pub request_id: Option<String>,
}

/// Endpoint-specific free-form fields carried by request envelopes.
pub type RequestExtra = serde_json::Map<String, serde_json::Value>;

/// Default maximum output token count for speech recognition.
pub const DEFAULT_TRANSCRIPTION_MAX_TOKENS: u32 = 512;

/// Text generation options shared by inference endpoints.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SippTextOptions {
    /// Maximum output tokens requested from the endpoint.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff.
    pub top_p: Option<f32>,
    /// Stop strings.
    pub stop: Vec<String>,
}

/// Local-only text generation options.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LocalTextOptions {
    /// Local KV-cache context key.
    pub context_key: Option<String>,
    /// Grammar constraint.
    pub grammar: Option<String>,
    /// JSON schema constraint.
    pub json_schema: Option<String>,
    /// Local runtime sampling override.
    pub sampling: Option<SamplingRuntimeOverride>,
    /// Binary media payloads for multimodal requests.
    pub media: Vec<Vec<u8>>,
}

impl LocalTextOptions {
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn has_fields(&self) -> bool {
        self.context_key.is_some()
            || self.grammar.is_some()
            || self.json_schema.is_some()
            || self.sampling.is_some()
            || !self.media.is_empty()
    }
}

/// Local-only embedding options.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LocalEmbedOptions {
    /// Local KV-cache context key.
    pub context_key: Option<String>,
    /// Whether to L2-normalize embeddings.
    pub normalize: Option<bool>,
}

impl LocalEmbedOptions {
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn has_fields(&self) -> bool {
        self.context_key.is_some() || self.normalize.is_some()
    }
}

/// Unified raw-prompt text request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SippQueryRequest {
    /// Target endpoint, or the single matching local endpoint when omitted.
    pub endpoint: Option<EndpointRef>,
    /// Raw prompt text.
    pub prompt: String,
    /// Shared text generation options.
    pub options: SippTextOptions,
    /// Local-only request options.
    pub local: LocalTextOptions,
    /// Extra fields interpreted by gateway and provider endpoints.
    pub extra: RequestExtra,
    /// Whether the returned run handle emits token batches.
    pub emit_tokens: bool,
}

/// Unified chat request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SippChatRequest {
    /// Target endpoint, or the single matching local endpoint when omitted.
    pub endpoint: Option<EndpointRef>,
    /// Chat messages.
    pub messages: Vec<ChatMessage>,
    /// Shared text generation options.
    pub options: SippTextOptions,
    /// Local-only request options.
    pub local: LocalTextOptions,
    /// Extra fields interpreted by gateway and provider endpoints.
    pub extra: RequestExtra,
    /// Whether the returned run handle emits token batches.
    pub emit_tokens: bool,
}

/// Unified single-input embedding request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SippEmbedRequest {
    /// Target endpoint, or the single matching local endpoint when omitted.
    pub endpoint: Option<EndpointRef>,
    /// Input text.
    pub input: String,
    /// Local-only embedding options.
    pub local: LocalEmbedOptions,
    /// Extra fields interpreted by gateway and provider endpoints.
    pub extra: RequestExtra,
}

/// Encoded-audio transcription request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SippListenRequest {
    /// Target endpoint, or the single matching local endpoint when omitted.
    pub endpoint: Option<EndpointRef>,
    /// Complete encoded WAV, MP3, or FLAC payload.
    pub audio: Vec<u8>,
    /// Optional language hint supplied to the local transcription model.
    pub language: Option<String>,
    /// Maximum number of transcript tokens to generate.
    pub max_tokens: Option<u32>,
}

impl SippListenRequest {
    /// Create a transcription request without a language hint.
    pub fn new(audio: impl Into<Vec<u8>>) -> Self {
        Self {
            endpoint: None,
            audio: audio.into(),
            language: None,
            max_tokens: None,
        }
    }

    /// Attach an exact language hint.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the maximum number of transcript tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl From<Vec<u8>> for SippListenRequest {
    fn from(audio: Vec<u8>) -> Self {
        Self::new(audio)
    }
}

impl From<&[u8]> for SippListenRequest {
    fn from(audio: &[u8]) -> Self {
        Self::new(audio)
    }
}

impl<const N: usize> From<[u8; N]> for SippListenRequest {
    fn from(audio: [u8; N]) -> Self {
        Self::new(audio)
    }
}

/// Text-to-WAV synthesis request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SippSpeakRequest {
    /// Target endpoint, or the single matching local endpoint when omitted.
    pub endpoint: Option<EndpointRef>,
    /// Text to synthesize.
    pub text: String,
    /// Optional language hint passed to the loaded synthesizer.
    pub language: Option<String>,
    /// Optional encoded WAV, MP3, or FLAC speaker reference.
    pub speaker_audio: Option<Vec<u8>>,
    /// Optional hard duration limit in milliseconds.
    pub max_duration_ms: Option<u32>,
}

impl SippSpeakRequest {
    /// Create a synthesis request using the loaded model's defaults.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            endpoint: None,
            text: text.into(),
            language: None,
            speaker_audio: None,
            max_duration_ms: None,
        }
    }

    /// Attach an exact language hint.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Attach encoded speaker-reference audio.
    pub fn speaker(mut self, audio: impl Into<Vec<u8>>) -> Self {
        self.speaker_audio = Some(audio.into());
        self
    }

    /// Set a hard duration limit; reaching it before end of generation fails.
    pub fn max_duration_ms(mut self, max_duration_ms: u32) -> Self {
        self.max_duration_ms = Some(max_duration_ms);
        self
    }
}

impl From<String> for SippSpeakRequest {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for SippSpeakRequest {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}
