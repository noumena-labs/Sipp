//! Provider adapters for CogentLM server and direct-provider integrations.

mod adapter_transport;
mod config;
mod error;
mod model;
mod providers;
mod request;
mod response;
mod stream;
mod transport;

pub use adapter_transport::{ProviderBackend, ProviderTransport};
pub use cogentlm_core::{CapabilitySupport, TokenUsage};
pub use config::{
    AnthropicAdapterConfig, OpenAiAdapterConfig, OpenAiCompatibleAdapterConfig,
    OpenAiCompatibleProtocol, ProviderAuth, ProviderKind, SecretString,
};
pub use error::{ProviderError, ProviderErrorKind, ProviderResult};
pub use model::{ProviderCapabilities, ProviderModel};
pub use providers::{AnthropicAdapter, OpenAiAdapter, OpenAiCompatibleAdapter};
pub use request::{
    ProviderChatRequest, ProviderEmbedRequest, ProviderGenerateRequest, ProviderGenerationOptions,
    ProviderOptions, ProviderRequestContext,
};
pub use response::{
    ProviderChatResponse, ProviderEmbeddingOutput, ProviderEmbeddingResponse,
    ProviderGenerateResponse, ProviderResponse, ProviderResponseMetadata, ProviderTextOutput,
};
pub use stream::{ProviderStream, ProviderStreamEvent};
pub(crate) use transport::{HttpByteStream, HttpTransport};
