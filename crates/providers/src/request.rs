use cogentlm_core::ChatMessage;

pub type ProviderOptions = serde_json::Map<String, serde_json::Value>;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProviderGenerationOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub options: ProviderGenerationOptions,
    pub provider_options: ProviderOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderGenerateRequest {
    pub model: String,
    pub prompt: String,
    pub options: ProviderGenerationOptions,
    pub provider_options: ProviderOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderEmbedRequest {
    pub model: String,
    pub input: String,
    pub provider_options: ProviderOptions,
}
