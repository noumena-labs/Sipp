mod config;
mod driver;
pub mod protocol;
mod token_emission;

pub use config::{
    CacheRuntimeConfig, ContextRuntimeConfig, FlashAttentionMode, GenerateOptions, GpuLayerConfig,
    KvCacheType, KvReuseMode, LogitBias, ModelPlacementConfig, MultimodalRuntimeConfig,
    NativeRuntimeConfig, ObservabilityRuntimeConfig, RequestSampling, ResidencyRuntimeConfig,
    ResolvedRuntimeLimits, RopeScaling, SamplerStage, SamplingRuntimeConfig, SamplingRuntimePatch,
    SchedulerRuntimeConfig, SplitMode, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};
pub use driver::{
    ChatMessage, ChatRequest, ChatRole, SippEngine, EngineEmbeddingResponseFuture,
    EngineEmbeddingRun, EngineEventReceiver, EngineLoad, EngineTextResponseFuture, EngineTextRun,
    EngineTokenBatches, QueryOptions, QueryRequest,
};
pub use protocol::{
    CacheSource, EmbedOptions, EmbedRequest, EmbeddingCapabilities, EmbeddingResult, EngineEvent,
    EngineState, EngineStats, FinishReason, GenerationResult, ModelCapabilities, ModelClass,
    PoolingType, RequestStats,
};
pub use token_emission::{TokenBatch, TokenEmissionStats};

#[cfg(test)]
#[path = "../tests/engine_tests.rs"]
mod engine_tests;
