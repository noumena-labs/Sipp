use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
