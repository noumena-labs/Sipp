mod backend;
mod chat;
pub mod engine;
mod error;
pub mod lifecycle;
pub mod runtime;
mod token;

pub use backend::{backend_observability_json, set_llama_log_quiet};
pub use chat::default_media_marker;
pub use engine::{
    CacheKeyPolicy, CacheRuntimeConfig, ChatMessage, ChatRequest, ChatRole, CogentEngine,
    ContextRuntimeConfig, EngineEvent, EngineEventReceiver, EngineState, EngineStats,
    FlashAttentionMode, GenerateOptions, GpuLayerConfig, KvCacheType, KvReuseMode, LogitBias,
    ModelPlacementConfig, MultimodalRuntimeConfig, NativeRuntimeConfig, ObservabilityRuntimeConfig,
    QueryOptions, QueryRequest, QueryResponse, RequestResult, ResidencyRuntimeConfig,
    ResolvedRuntimeLimits, RopeScaling, SamplerStage, SamplingRuntimeConfig,
    SchedulerRuntimeConfig, SplitMode, StreamStats, TokenBatch, TokenFrame, TokenStreamMode,
};
pub use error::{Error, Result};
pub use lifecycle::{
    model_entry_from_assets, model_source_from_path, read_backend_capabilities,
    vision_model_source_from_paths, AssetInspection, AssetInstallResult, AssetRecord, AssetRole,
    AssetSource, AssetStore, BackendCapabilities, BackendPlan, BackendPolicy, BackendPreference,
    BackendSelection, ClassifiedAsset, GgufMetadataInspection, LoadedModelInfo,
    LocalStorageBackend, ModelAsset, ModelAssetKind, ModelAssets, ModelDetection,
    ModelDetectionMethod, ModelEntry, ModelError, ModelInfo, ModelLoadOptions, ModelModality,
    ModelPairing, ModelPairingReason, ModelPairingState, ModelRegistry, ModelService,
    ModelServiceState, ModelSource, ModelSourceKind, ModelStatus, PairingPlan, PairingResolver,
    RegistryManifest, RemovedModel, StatsMode, StorageBackend,
};
pub use token::{token_to_piece, tokenize};
