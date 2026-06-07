use cogentlm_core::CapabilitySupport;

use crate::ProviderKind;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub chat: CapabilitySupport,
    pub generate: CapabilitySupport,
    pub embeddings: CapabilitySupport,
    pub token_emission: CapabilitySupport,
}

impl ProviderCapabilities {
    pub const fn unknown() -> Self {
        Self {
            chat: CapabilitySupport::Unknown,
            generate: CapabilitySupport::Unknown,
            embeddings: CapabilitySupport::Unknown,
            token_emission: CapabilitySupport::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderModel {
    pub id: String,
    pub provider: ProviderKind,
    pub display_name: Option<String>,
    pub capabilities: ProviderCapabilities,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub raw: serde_json::Value,
}
