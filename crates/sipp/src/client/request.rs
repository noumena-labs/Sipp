use crate::core::ChatMessage;
use crate::engine::SamplingRuntimeConfig;

use crate::client::EndpointRef;

/// Request-scoped metadata propagated through endpoint execution.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SippRequestContext {
    /// Canonical request identifier assigned by the application boundary.
    pub request_id: Option<String>,
}

/// Endpoint-specific free-form options carried by request envelopes.
pub type EndpointOptions = serde_json::Map<String, serde_json::Value>;

/// Direct provider free-form options carried by request envelopes.
pub type ProviderOptions = serde_json::Map<String, serde_json::Value>;

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
    pub sampling: Option<SamplingRuntimeConfig>,
    /// Binary media payloads for multimodal requests.
    pub media: Vec<Vec<u8>>,
}

impl LocalTextOptions {
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
    /// Endpoint-specific options passed to gateway endpoint implementations.
    pub endpoint_options: EndpointOptions,
    /// Direct-provider-only request options passed to provider adapters.
    pub provider_options: ProviderOptions,
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
    /// Endpoint-specific options passed to gateway endpoint implementations.
    pub endpoint_options: EndpointOptions,
    /// Direct-provider-only request options passed to provider adapters.
    pub provider_options: ProviderOptions,
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
    /// Endpoint-specific options passed to gateway endpoint implementations.
    pub endpoint_options: EndpointOptions,
    /// Direct-provider-only request options passed to provider adapters.
    pub provider_options: ProviderOptions,
}
