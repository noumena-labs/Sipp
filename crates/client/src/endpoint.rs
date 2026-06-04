use cogentlm_core::CapabilitySupport;
use cogentlm_engine::engine::ModelCapabilities;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/endpoint_tests.rs"]
mod endpoint_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Addressable inference destination.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EndpointRef {
    /// Local engine endpoint registered in the client.
    Local {
        /// Client-scoped endpoint id.
        id: String,
    },
    /// Remote endpoint registered in the client.
    Remote {
        /// Client-scoped endpoint id.
        id: String,
    },
}

impl EndpointRef {
    pub(crate) fn is_local(&self) -> bool {
        matches!(self, Self::Local { .. })
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

    #[cfg(feature = "remote")]
    pub(crate) const fn unknown() -> Self {
        Self {
            query: CapabilitySupport::Unknown,
            chat: CapabilitySupport::Unknown,
            embed: CapabilitySupport::Unknown,
        }
    }

    pub(crate) fn for_operation(&self, operation: &'static str) -> CapabilitySupport {
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
