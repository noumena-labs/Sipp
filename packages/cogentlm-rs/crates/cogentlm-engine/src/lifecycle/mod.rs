mod backend_policy;
mod browser_catalog;
mod gguf;
mod pairing;
mod registry;
mod service;
mod storage;
mod types;
mod util;

pub use backend_policy::{
    read_backend_capabilities, BackendCapabilities, BackendPlan, BackendPolicy,
};
pub use browser_catalog::{
    error_response as browser_lifecycle_error_response,
    response_json as browser_lifecycle_response_json,
    success_response as browser_lifecycle_success_response, BrowserAssetRecord,
    BrowserCommitLoadRequest, BrowserCommitLoadResponse, BrowserCreateConfig,
    BrowserLifecycleEnvelope, BrowserLifecycleError, BrowserLifecycleService,
    BrowserLifecycleState, BrowserLoadOptions, BrowserLoadSource, BrowserModelEntry,
    BrowserModelInfo, BrowserObservabilityEvent, BrowserObservabilityEventType,
    BrowserObservabilityMode, BrowserObservabilitySnapshot, BrowserPlannedAsset,
    BrowserPrepareLoadResponse, BrowserQueryObservation, BrowserRegistryManifest,
    BrowserRemoveResponse,
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
