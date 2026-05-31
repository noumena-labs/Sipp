//! Provider interaction API for remote and proxy model services.

mod config;
mod error;
mod model;
mod provider_transport;
mod providers;
mod request;
mod response;
mod stream;
mod transport;

pub use cogentlm_core::{CapabilitySupport, TokenUsage};
pub use config::{
    AnthropicConfig, OpenAiConfig, ProviderAuth, ProviderKind, ProxyConfig, ProxyProtocol,
    SecretString,
};
pub use error::{ProviderError, ProviderErrorKind, ProviderResult};
pub use model::{ProviderCapabilities, ProviderModel};
pub use provider_transport::{ProviderBackend, ProviderTransport};
pub use providers::{AnthropicProvider, OpenAiProvider, ProxyProvider};
pub use request::{
    ProviderChatRequest, ProviderEmbedRequest, ProviderGenerateRequest, ProviderGenerationOptions,
    ProviderOptions,
};
pub use response::{
    ProviderChatResponse, ProviderEmbeddingOutput, ProviderEmbeddingResponse,
    ProviderGenerateResponse, ProviderResponse, ProviderResponseMetadata, ProviderTextOutput,
};
pub use stream::{ProviderStream, ProviderStreamEvent};
pub(crate) use transport::{HttpByteStream, HttpTransport};
