use crate::core::{CapabilitySupport, Operation};
use crate::engine::{ModelCapabilities, NativeRuntimeConfig};
use crate::lifecycle::ManagedModel;

use crate::client::GatewayDescriptor;
#[cfg(feature = "providers")]
use crate::client::ProviderDescriptor;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/client/endpoint_tests.rs"]
mod endpoint_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Configuration consumed by `SippClient::add` to register an endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Endpoint {
    pub(super) kind: EndpointKind,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum EndpointKind {
    Local(Box<Local>),
    Gateway(Box<GatewayDescriptor>),
    #[cfg(feature = "providers")]
    Provider(Box<ProviderDescriptor>),
}

impl From<Local> for Endpoint {
    fn from(local: Local) -> Self {
        Self {
            kind: EndpointKind::Local(Box::new(local)),
        }
    }
}

impl From<GatewayDescriptor> for Endpoint {
    fn from(gateway: GatewayDescriptor) -> Self {
        Self {
            kind: EndpointKind::Gateway(Box::new(gateway)),
        }
    }
}

#[cfg(feature = "providers")]
impl From<ProviderDescriptor> for Endpoint {
    fn from(provider: ProviderDescriptor) -> Self {
        Self {
            kind: EndpointKind::Provider(Box::new(provider)),
        }
    }
}

/// Local endpoint input for a model managed by the client model store.
#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub(super) model_id: String,
    pub(super) runtime: NativeRuntimeConfig,
}

impl Local {
    /// Create a local endpoint input using the canonical runtime configuration.
    pub fn new(model: &ManagedModel) -> Self {
        Self {
            model_id: model.id.clone(),
            runtime: NativeRuntimeConfig::default(),
        }
    }

    /// Set the native runtime configuration used when loading the model.
    #[must_use]
    pub fn runtime(mut self, runtime: NativeRuntimeConfig) -> Self {
        self.runtime = runtime;
        self
    }
}

/// Addressable inference destination.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EndpointRef {
    id: String,
}

impl EndpointRef {
    /// Return the stable client-scoped endpoint id.
    pub fn id(&self) -> &str {
        &self.id
    }

    #[doc(hidden)]
    pub fn from_id(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

/// Cached support for the public inference operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndpointCapabilities {
    /// Raw-prompt text generation support.
    pub query: CapabilitySupport,
    /// Chat generation support.
    pub chat: CapabilitySupport,
    /// Embedding support.
    pub embed: CapabilitySupport,
    /// Encoded-audio transcription support.
    pub listen: CapabilitySupport,
    /// Text-to-WAV synthesis support.
    pub speak: CapabilitySupport,
}

impl EndpointCapabilities {
    pub(crate) fn from_local(capabilities: &ModelCapabilities) -> Self {
        Self {
            query: support(capabilities.supports_operation(Operation::Query)),
            chat: support(capabilities.supports_operation(Operation::Chat)),
            embed: support(capabilities.supports_operation(Operation::Embed)),
            listen: support(capabilities.supports_operation(Operation::Listen)),
            speak: support(capabilities.supports_operation(Operation::Speak)),
        }
    }

    /// Return runtime-determined remote text capabilities with no speech support.
    pub const fn remote_text() -> Self {
        Self {
            query: CapabilitySupport::Unknown,
            chat: CapabilitySupport::Unknown,
            embed: CapabilitySupport::Unknown,
            listen: CapabilitySupport::Unsupported,
            speak: CapabilitySupport::Unsupported,
        }
    }

    /// Return support for one canonical inference operation.
    pub const fn for_operation(&self, operation: Operation) -> CapabilitySupport {
        match operation {
            Operation::Query => self.query,
            Operation::Chat => self.chat,
            Operation::Embed => self.embed,
            Operation::Listen => self.listen,
            Operation::Speak => self.speak,
        }
    }
}

const fn support(enabled: bool) -> CapabilitySupport {
    if enabled {
        CapabilitySupport::Supported
    } else {
        CapabilitySupport::Unsupported
    }
}
