//! GGUF metadata inspection re-exported through the engine lifecycle API.

use std::path::Path;

use super::types::{GgufMetadataInspection, ModelDetection, ModelError};

pub fn inspect_gguf_metadata_path(
    path: impl AsRef<Path>,
) -> Result<Option<GgufMetadataInspection>, ModelError> {
    cogentlm_shard::inspect_gguf_metadata_path(path).map_err(ModelError::from)
}

pub fn inspect_gguf_metadata(bytes: &[u8]) -> Result<Option<GgufMetadataInspection>, ModelError> {
    cogentlm_shard::inspect_gguf_metadata(bytes).map_err(ModelError::from)
}

pub fn detect_model_from_gguf_bytes(
    name: impl Into<String>,
    bytes: &[u8],
) -> Result<ModelDetection, ModelError> {
    cogentlm_shard::detect_model_from_gguf_bytes(name, bytes).map_err(ModelError::from)
}
