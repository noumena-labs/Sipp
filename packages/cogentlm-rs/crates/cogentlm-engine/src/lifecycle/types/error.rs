use thiserror::Error as ThisError;

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

impl From<cogentlm_shard::GgufError> for ModelError {
    fn from(error: cogentlm_shard::GgufError) -> Self {
        match error {
            cogentlm_shard::GgufError::Io(error) => Self::Io(error),
            cogentlm_shard::GgufError::Invalid(message) => Self::InvalidGgufMetadata(message),
            cogentlm_shard::GgufError::UnsupportedVersion(version) => {
                Self::UnsupportedGgufVersion(version)
            }
            cogentlm_shard::GgufError::MetadataTooLarge { max_bytes } => {
                Self::GgufMetadataTooLarge { max_bytes }
            }
            cogentlm_shard::GgufError::AlreadySplit(count) => Self::InvalidGgufMetadata(format!(
                "source GGUF is already split into {count} files"
            )),
        }
    }
}
