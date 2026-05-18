mod backend_policy;
mod gguf;
mod pairing;
mod registry;
mod service;
mod storage;
mod types;

pub use backend_policy::{
    read_backend_capabilities, BackendCapabilities, BackendPlan, BackendPolicy,
};
pub use gguf::{detect_model_from_gguf_bytes, inspect_gguf_metadata, inspect_gguf_metadata_path};
pub use pairing::PairingResolver;
pub use registry::{model_entry_from_assets, ModelRegistry, RemovedModel};
pub use service::{
    model_source_from_path, vision_model_source_from_paths, LoadedModelInfo, ModelService,
};
pub use storage::{AssetInstallResult, AssetStore, LocalStorageBackend, StorageBackend};
pub use types::{
    AssetInspection, AssetRecord, AssetRole, AssetSource, BackendPreference, BackendSelection,
    ClassifiedAsset, GgufMetadataInspection, ModelAsset, ModelAssetKind, ModelAssets,
    ModelDetection, ModelDetectionMethod, ModelEntry, ModelError, ModelInfo, ModelLoadOptions,
    ModelModality, ModelPairing, ModelPairingReason, ModelPairingState, ModelServiceState,
    ModelSource, ModelSourceKind, ModelStatus, PairingPlan, RegistryManifest, StatsMode,
};
