//! Shared lifecycle types, grouped by lifecycle concern.

mod assets;
mod error;
mod model;
mod runtime;

pub use crate::shard::{GgufMetadataInspection, ModelDetection, ModelDetectionMethod};
pub use assets::{AssetInspection, AssetRecord, AssetRole, AssetSource, ModelAssetKind};
pub use error::ModelError;
pub use model::{
    ClassifiedAsset, ModelAsset, ModelAssets, ModelEntry, ModelInfo, ModelModality, ModelPairing,
    ModelPairingReason, ModelPairingState, ModelSource, ModelSourceKind, ModelStatus, PairingPlan,
    RegistryManifest, REGISTRY_MANIFEST_VERSION,
};
pub use runtime::{
    BackendPreference, BackendSelection, ModelLoadOptions, ModelServiceState, StatsMode,
    DEFAULT_MODEL_BACKEND, DEFAULT_MODEL_STATS,
};

#[cfg(test)]
#[path = "../../tests/lifecycle/types_tests.rs"]
mod types_tests;
