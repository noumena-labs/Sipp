use cogentlm_core::CapabilitySupport;
use cogentlm_engine::engine::ModelCapabilities;

/// Addressable inference destination.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EndpointRef {
    LocalEngine { engine: String },
    ProviderModel { provider: String, model: String },
}

impl EndpointRef {
    pub(crate) fn is_local_engine(&self) -> bool {
        matches!(self, Self::LocalEngine { .. })
    }
}

/// Cached support for the three public inference verbs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndpointCapabilities {
    pub query: CapabilitySupport,
    pub chat: CapabilitySupport,
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

    #[cfg(feature = "providers")]
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
