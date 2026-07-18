use crate::client::GatewayDescriptor;
#[cfg(feature = "providers")]
use crate::client::ProviderDescriptor;
use crate::engine::NativeRuntimeConfig;

/// Default native registry and managed-asset root for a client.
pub const DEFAULT_STORAGE_ROOT: &str = ".sipp-models";

/// Descriptor used by `SippClient::add` to register an endpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum EndpointDescriptor {
    /// Local GGUF model loaded into this process.
    Local(LocalDescriptor),
    /// First-party HTTP gateway endpoint.
    Gateway(GatewayDescriptor),
    /// Direct provider endpoint using caller-owned credentials.
    #[cfg(feature = "providers")]
    Provider(ProviderDescriptor),
}

impl From<LocalDescriptor> for EndpointDescriptor {
    fn from(descriptor: LocalDescriptor) -> Self {
        Self::Local(descriptor)
    }
}

impl From<GatewayDescriptor> for EndpointDescriptor {
    fn from(descriptor: GatewayDescriptor) -> Self {
        Self::Gateway(descriptor)
    }
}

#[cfg(feature = "providers")]
impl From<ProviderDescriptor> for EndpointDescriptor {
    fn from(descriptor: ProviderDescriptor) -> Self {
        Self::Provider(descriptor)
    }
}

/// Descriptor for a local GGUF model endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalDescriptor {
    /// Model ID returned by the client model store.
    pub model_id: String,
    /// Native runtime configuration.
    pub runtime: NativeRuntimeConfig,
}

impl LocalDescriptor {
    /// Create a local endpoint for a model in the client model store.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            runtime: NativeRuntimeConfig::default(),
        }
    }
}
