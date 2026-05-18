use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

use crate::engine::protocol::{BackendInfo, EngineStatus, RequestState};
use crate::engine::{EngineStats, NativeRuntimeConfig, ResolvedRuntimeLimits};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatsMode {
    Off,
    #[default]
    Basic,
    Profile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    WebGpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendSelection {
    pub requested: BackendPreference,
    pub selected: String,
    pub available: Vec<String>,
    pub gpu_offload_expected: bool,
    pub reason: Option<String>,
}

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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelLoadOptions {
    pub backend: BackendPreference,
    pub stats: StatsMode,
    pub runtime: NativeRuntimeConfig,
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
pub enum ModelAssetKind {
    Model,
    Projector,
    Shard,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssetSource {
    Local {
        path: PathBuf,
        modified_unix_ms: Option<u64>,
    },
    Remote {
        url: String,
        etag: Option<String>,
        last_modified: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetRole {
    Model,
    Projector,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetInspection {
    pub version: u32,
    pub role: AssetRole,
    pub architecture: Option<String>,
    pub vision_capable: bool,
    pub compatible_vision_projector_types: Vec<String>,
    pub provided_vision_projector_type: Option<String>,
}

impl AssetInspection {
    pub fn unknown() -> Self {
        Self {
            version: 1,
            role: AssetRole::Unknown,
            architecture: None,
            vision_capable: false,
            compatible_vision_projector_types: Vec::new(),
            provided_vision_projector_type: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetRecord {
    pub id: String,
    pub kind: ModelAssetKind,
    pub name: String,
    pub hash: String,
    pub bytes: u64,
    pub storage_path: PathBuf,
    pub source: AssetSource,
    pub ref_count: u32,
    pub created_at_unix_ms: u64,
    pub inspection: Option<AssetInspection>,
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
pub struct GgufMetadataInspection {
    pub general_type: Option<String>,
    pub general_architecture: Option<String>,
    pub clip_projector_type: Option<String>,
    pub clip_vision_projector_type: Option<String>,
    pub clip_has_vision_encoder: Option<bool>,
    pub scanned_key_count: usize,
    pub stopped_early_at_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDetection {
    pub inspection: AssetInspection,
    pub detection_method: ModelDetectionMethod,
    pub model_name: String,
    pub model_type: Option<String>,
    pub model_architecture: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDetectionMethod {
    GgufMetadata,
    None,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ModelServiceState {
    pub status: EngineStatus,
    pub model: Option<ModelInfo>,
    pub backend: BackendInfo,
    pub runtime: Option<ResolvedRuntimeLimits>,
    pub requests: Vec<RequestState>,
    pub stats: EngineStats,
    pub updated_at_unix_ms: u64,
}

impl Default for ModelServiceState {
    fn default() -> Self {
        Self {
            status: EngineStatus::Idle,
            model: None,
            backend: BackendInfo::default(),
            runtime: None,
            requests: Vec::new(),
            stats: EngineStats::default(),
            updated_at_unix_ms: 0,
        }
    }
}

#[derive(Debug, ThisError)]
pub enum ModelError {
    #[error("invalid model source: {0}")]
    InvalidModelSource(String),

    #[error("invalid model pairing: {0}")]
    InvalidModelPairing(String),

    #[error("unsupported GGUF version {0}")]
    UnsupportedGgufVersion(u32),

    #[error("invalid GGUF metadata: {0}")]
    InvalidGgufMetadata(String),

    #[error("GGUF metadata prefix exceeded {max_bytes} bytes")]
    GgufMetadataTooLarge { max_bytes: usize },

    #[error("model storage unavailable: {0}")]
    StorageUnavailable(String),

    #[error("model storage is corrupt: {0}")]
    StorageCorrupt(String),

    #[error("model asset is missing or corrupt: {0}")]
    AssetMissing(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("remote model loading is not available in this runtime: {0}")]
    RemoteUnavailable(String),

    #[error("model runtime failed: {0}")]
    Runtime(String),

    #[error("model registry JSON failed: {0}")]
    RegistryJson(#[from] serde_json::Error),

    #[error("model IO failed: {0}")]
    Io(#[from] std::io::Error),
}

impl From<crate::Error> for ModelError {
    fn from(error: crate::Error) -> Self {
        Self::Runtime(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_asset_source_requires_source_path() {
        let error = serde_json::from_str::<AssetSource>(r#"{"kind":"local"}"#)
            .expect_err("local source without path should be rejected");

        assert!(error.to_string().contains("missing field `path`"));
    }
}
