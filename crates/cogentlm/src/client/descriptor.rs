use std::path::PathBuf;

use crate::engine::NativeRuntimeConfig;

use crate::client::GatewayEndpointConfig;
#[cfg(feature = "providers")]
use crate::client::ProviderEndpointConfig;

/// Configuration used by `CogentClient::add` to register an endpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum EndpointDescriptor {
    /// Local GGUF model loaded into this process.
    LocalModel(LocalModelDescriptor),
    /// First-party HTTP gateway endpoint.
    Gateway(GatewayEndpointConfig),
    /// Direct provider endpoint using caller-owned credentials.
    #[cfg(feature = "providers")]
    Provider(ProviderEndpointConfig),
}

impl EndpointDescriptor {
    /// Create a local model descriptor.
    pub fn local(model_path: impl Into<PathBuf>, config: NativeRuntimeConfig) -> Self {
        Self::LocalModel(LocalModelDescriptor {
            model_path: model_path.into(),
            config,
        })
    }

    /// Create a gateway endpoint descriptor.
    pub fn gateway(config: GatewayEndpointConfig) -> Self {
        Self::Gateway(config)
    }

    /// Create a direct provider descriptor.
    #[cfg(feature = "providers")]
    pub fn provider(config: ProviderEndpointConfig) -> Self {
        Self::Provider(config)
    }
}

/// Local GGUF model descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalModelDescriptor {
    /// Path to the local model artifact.
    pub model_path: PathBuf,
    /// Native runtime configuration.
    pub config: NativeRuntimeConfig,
}
