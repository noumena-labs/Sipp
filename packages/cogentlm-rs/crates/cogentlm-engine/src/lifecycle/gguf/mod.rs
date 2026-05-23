//! GGUF metadata inspection through the engine lifecycle API.

use super::types::{ModelDetection, ModelError};

pub fn detect_model_from_gguf_bytes(
    name: impl Into<String>,
    bytes: &[u8],
) -> Result<ModelDetection, ModelError> {
    cogentlm_shard::detect_model_from_gguf_bytes(name, bytes).map_err(ModelError::from)
}
