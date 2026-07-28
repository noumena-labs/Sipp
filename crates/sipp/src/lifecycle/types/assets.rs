use std::path::PathBuf;

pub use crate::shard::{AssetInspection, AssetRole};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAssetKind {
    Model,
    Projector,
    Shard,
}

/// Base used to resolve a persisted local model path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPathAnchor {
    /// The persisted path is resolved directly by the operating system.
    #[default]
    Absolute,
    /// The persisted path is relative to the configured local source root.
    SourceRoot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssetSource {
    Local {
        path: PathBuf,
        #[serde(default)]
        anchor: LocalPathAnchor,
        modified_unix_ms: Option<u64>,
    },
    Remote {
        url: String,
        etag: Option<String>,
        last_modified: Option<String>,
    },
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

#[cfg(test)]
#[path = "../../tests/lifecycle/types/assets_tests.rs"]
mod assets_tests;
