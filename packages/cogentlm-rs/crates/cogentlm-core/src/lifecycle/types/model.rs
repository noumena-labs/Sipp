use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::assets::{AssetInspection, AssetRecord};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAsset {
    Path { path: PathBuf },
    Url { url: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAssets {
    Path { path: PathBuf },
    Paths { paths: Vec<PathBuf> },
    Url { url: String },
    Urls { urls: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    Installed {
        id: String,
    },
    Assets {
        model: ModelAssets,
        projector: Option<ModelAsset>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelModality {
    Text,
    Vision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Ready,
    NeedsProjector,
    Broken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    Local,
    Remote,
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
            version: 3,
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
