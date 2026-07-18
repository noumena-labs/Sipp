use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::assets::{AssetInspection, AssetRecord};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/types/model_tests.rs"]
mod model_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub const REGISTRY_MANIFEST_VERSION: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelModality {
    Text,
    Vision,
}

impl ModelModality {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Ready,
    NeedsProjector,
    Broken,
}

impl ModelStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NeedsProjector => "needs_projector",
            Self::Broken => "broken",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    Local,
    Remote,
}

impl ModelSourceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub modality: ModelModality,
    pub status: ModelStatus,
    pub source: ModelSourceKind,
    pub bytes: u64,
    pub loaded: bool,
    pub chat_template: Option<String>,
    pub bos_text: String,
    pub eos_text: String,
    pub media_marker: Option<String>,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
}

/// Model managed by a client model store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManagedModel {
    /// Stable id used to create local endpoint descriptors.
    pub id: String,
    /// Display name read from model metadata or the source filename.
    pub name: String,
    /// Total bytes occupied by the model and optional projector.
    pub bytes: u64,
    /// Inference modality detected from model metadata.
    pub modality: ModelModality,
    /// Current installation and pairing state.
    pub status: ModelStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPairingState {
    Resolved,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelPairingReason {
    BaseNotVision,
    NoMatch,
    MultipleMatches,
    MissingMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPairing {
    pub state: ModelPairingState,
    pub checked_projector_index_revision: u64,
    pub compatible_vision_projector_types: Vec<String>,
    pub reason: Option<ModelPairingReason>,
    pub updated_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub modality: ModelModality,
    pub status: ModelStatus,
    pub model_asset_ids: Vec<String>,
    pub projector_asset_id: Option<String>,
    pub pairing: Option<ModelPairing>,
    pub runtime_fingerprint: Option<String>,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
    pub last_loaded_at_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegistryManifest {
    pub version: u32,
    pub projector_index_revision: u64,
    pub assets: BTreeMap<String, AssetRecord>,
    pub models: BTreeMap<String, ModelEntry>,
}

impl Default for RegistryManifest {
    fn default() -> Self {
        Self {
            version: REGISTRY_MANIFEST_VERSION,
            projector_index_revision: 0,
            assets: BTreeMap::new(),
            models: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifiedAsset {
    pub asset_id: String,
    pub name: String,
    pub inspection: AssetInspection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PairingPlan {
    pub model_asset_ids: Vec<String>,
    pub projector_asset_id: Option<String>,
    pub name: String,
    pub modality: ModelModality,
    pub status: ModelStatus,
    pub compatible_vision_projector_types: Vec<String>,
}
