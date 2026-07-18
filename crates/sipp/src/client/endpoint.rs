use crate::core::CapabilitySupport;
use crate::engine::ModelCapabilities;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/client/endpoint_tests.rs"]
mod endpoint_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

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

/// Cached support for the three public inference verbs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndpointCapabilities {
    /// Raw-prompt text generation support.
    pub query: CapabilitySupport,
    /// Chat generation support.
    pub chat: CapabilitySupport,
    /// Embedding support.
    pub embed: CapabilitySupport,
}

impl EndpointCapabilities {
    pub(crate) fn from_local(capabilities: &ModelCapabilities) -> Self {
        Self {
            query: support(capabilities.supports_text_generation),
            chat: support(capabilities.supports_text_generation && capabilities.has_chat_template),
            embed: support(capabilities.supports_embeddings),
        }
    }

    /// Return capabilities that will be determined by the endpoint at runtime.
    pub const fn unknown() -> Self {
        Self {
            query: CapabilitySupport::Unknown,
            chat: CapabilitySupport::Unknown,
            embed: CapabilitySupport::Unknown,
        }
    }

    /// Return support for one canonical inference operation.
    pub fn for_operation(&self, operation: &'static str) -> CapabilitySupport {
        match operation {
            "query" => self.query,
            "chat" => self.chat,
            "embed" => self.embed,
            _ => CapabilitySupport::Unsupported,
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
