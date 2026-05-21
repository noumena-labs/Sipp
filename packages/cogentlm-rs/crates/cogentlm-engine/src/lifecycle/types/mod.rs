//! Shared lifecycle types, grouped by lifecycle concern.

mod assets;
mod detection;
mod error;
mod model;
mod runtime;

pub use assets::{AssetInspection, AssetRecord, AssetRole, AssetSource, ModelAssetKind};
pub use detection::{GgufMetadataInspection, ModelDetection, ModelDetectionMethod};
pub use error::ModelError;
pub use model::{
    ClassifiedAsset, ModelAsset, ModelAssets, ModelEntry, ModelInfo, ModelModality, ModelPairing,
    ModelPairingReason, ModelPairingState, ModelSource, ModelSourceKind, ModelStatus, PairingPlan,
    RegistryManifest,
};
pub use runtime::{
    BackendPreference, BackendSelection, ModelLoadOptions, ModelServiceState, StatsMode,
};

#[cfg(test)]
mod tests {
    mod types_tests;
}
