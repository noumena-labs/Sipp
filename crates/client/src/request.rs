use cogentlm_core::ChatMessage;
use cogentlm_engine::engine::SamplingRuntimeConfig;

use crate::EndpointRef;

/// Provider-neutral free-form options carried by request envelopes.
pub type ProviderOptions = serde_json::Map<String, serde_json::Value>;

/// Text generation options shared by local and provider endpoints.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CogentTextOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Vec<String>,
}

/// Local-only text generation options.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LocalTextOptions {
    pub context_key: Option<String>,
    pub grammar: Option<String>,
    pub json_schema: Option<String>,
    pub sampling: Option<SamplingRuntimeConfig>,
    pub media: Vec<Vec<u8>>,
}

impl LocalTextOptions {
    #[cfg(feature = "providers")]
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
    pub context_key: Option<String>,
    pub normalize: Option<bool>,
}

impl LocalEmbedOptions {
    #[cfg(feature = "providers")]
    pub(crate) fn has_fields(&self) -> bool {
        self.context_key.is_some() || self.normalize.is_some()
    }
}

/// Unified raw-prompt text request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CogentQueryRequest {
    pub endpoint: Option<EndpointRef>,
    pub prompt: String,
    pub options: CogentTextOptions,
    pub local: LocalTextOptions,
    pub provider_options: ProviderOptions,
    pub stream_tokens: bool,
}

/// Unified chat request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CogentChatRequest {
    pub endpoint: Option<EndpointRef>,
    pub messages: Vec<ChatMessage>,
    pub options: CogentTextOptions,
    pub local: LocalTextOptions,
    pub provider_options: ProviderOptions,
    pub stream_tokens: bool,
}

/// Unified single-input embedding request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CogentEmbedRequest {
    pub endpoint: Option<EndpointRef>,
    pub input: String,
    pub local: LocalEmbedOptions,
    pub provider_options: ProviderOptions,
}
