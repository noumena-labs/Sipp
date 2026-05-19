use serde::{Deserialize, Serialize};

use super::assets::AssetInspection;

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
