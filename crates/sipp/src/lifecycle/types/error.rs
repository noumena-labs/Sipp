use thiserror::Error as ThisError;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/types/error_tests.rs"]
mod error_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

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

    #[error("model is in use: {0}")]
    ModelInUse(String),

    #[error("failed to initialize remote model acquisition: {0}")]
    RemoteClient(String),

    #[error("remote metadata is unavailable for {url}: {reason}")]
    RemoteMetadataUnavailable {
        url: String,
        status: Option<u16>,
        retry_after_ms: Option<u64>,
        reason: String,
    },

    #[error("remote model download failed for {url}: {reason}")]
    RemoteDownloadFailed {
        url: String,
        status: Option<u16>,
        retry_after_ms: Option<u64>,
        reason: String,
    },

    #[error("remote model integrity failed for {url}: {reason}")]
    RemoteIntegrityFailed { url: String, reason: String },

    #[error("remote model cleanup failed for {url}: {reason}")]
    RemoteCleanupFailed { url: String, reason: String },

    #[error("remote acquisition was cancelled")]
    AcquisitionCancelled,

    #[error("stale remote acquisition result: expected {expected}, received {received}")]
    StaleAcquisitionResult { expected: String, received: String },

    #[error("model runtime failed: {0}")]
    Runtime(String),

    #[error("unsupported operation {operation}: {reason}")]
    UnsupportedOperation {
        operation: &'static str,
        reason: String,
    },

    #[error("model registry JSON failed: {0}")]
    RegistryJson(#[from] serde_json::Error),

    #[error("model IO failed: {0}")]
    Io(#[from] std::io::Error),
}

impl ModelError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidModelSource(_)
            | Self::UnsupportedGgufVersion(_)
            | Self::InvalidGgufMetadata(_)
            | Self::GgufMetadataTooLarge { .. } => "INVALID_MODEL_SOURCE",
            Self::InvalidModelPairing(_) => "INVALID_MODEL_PAIRING",
            Self::StorageUnavailable(_) | Self::Io(_) => "STORAGE_UNAVAILABLE",
            Self::StorageCorrupt(_) | Self::RegistryJson(_) => "STORAGE_CORRUPT",
            Self::AssetMissing(_) => "MODEL_BROKEN",
            Self::ModelNotFound(_) => "MODEL_NOT_FOUND",
            Self::ModelInUse(_) => "MODEL_IN_USE",
            Self::RemoteMetadataUnavailable { .. } => "REMOTE_METADATA_UNAVAILABLE",
            Self::RemoteClient(_)
            | Self::RemoteDownloadFailed { .. }
            | Self::RemoteIntegrityFailed { .. }
            | Self::RemoteCleanupFailed { .. } => "REMOTE_LOAD_FAILED",
            Self::AcquisitionCancelled => "ACQUISITION_CANCELLED",
            Self::StaleAcquisitionResult { .. } => "STALE_ACQUISITION_RESULT",
            Self::Runtime(_) => "QUERY_FAILED",
            Self::UnsupportedOperation { .. } => "UNSUPPORTED_OPERATION",
        }
    }

    pub const fn status(&self) -> Option<u16> {
        match self {
            Self::RemoteMetadataUnavailable { status, .. }
            | Self::RemoteDownloadFailed { status, .. } => *status,
            _ => None,
        }
    }

    pub const fn retry_after_ms(&self) -> Option<u64> {
        match self {
            Self::RemoteMetadataUnavailable { retry_after_ms, .. }
            | Self::RemoteDownloadFailed { retry_after_ms, .. } => *retry_after_ms,
            _ => None,
        }
    }
}

impl From<crate::error::Error> for ModelError {
    fn from(error: crate::error::Error) -> Self {
        match error {
            crate::error::Error::UnsupportedOperation { operation, reason } => {
                Self::UnsupportedOperation { operation, reason }
            }
            error => Self::Runtime(error.to_string()),
        }
    }
}

impl From<crate::shard::GgufError> for ModelError {
    fn from(error: crate::shard::GgufError) -> Self {
        match error {
            crate::shard::GgufError::Io(error) => Self::Io(error),
            crate::shard::GgufError::Invalid(message) => Self::InvalidGgufMetadata(message),
            crate::shard::GgufError::UnsupportedVersion(version) => {
                Self::UnsupportedGgufVersion(version)
            }
            crate::shard::GgufError::MetadataTooLarge { max_bytes } => {
                Self::GgufMetadataTooLarge { max_bytes }
            }
            crate::shard::GgufError::AlreadySplit(count) => Self::InvalidGgufMetadata(format!(
                "source GGUF is already split into {count} files"
            )),
        }
    }
}
