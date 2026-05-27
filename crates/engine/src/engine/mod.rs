mod config;
mod driver;
pub mod protocol;
mod stream;

pub use config::{
    CacheKeyPolicy, CacheRuntimeConfig, ContextRuntimeConfig, FlashAttentionMode, GenerateOptions,
    GpuLayerConfig, KvCacheType, KvReuseMode, LogitBias, ModelPlacementConfig,
    MultimodalRuntimeConfig, NativeRuntimeConfig, ObservabilityRuntimeConfig,
    ResidencyRuntimeConfig, ResolvedRuntimeLimits, RopeScaling, SamplerStage,
    SamplingRuntimeConfig, SchedulerRuntimeConfig, SplitMode, DEFAULT_CONTEXT_KEY,
    DEFAULT_MAX_TOKENS,
};
pub use driver::{
    ChatMessage, ChatRequest, ChatRole, CogentEngine, EngineEventReceiver, QueryOptions,
    QueryRequest,
};
pub use protocol::{
    EmbedOptions, EmbedRequest, EmbeddingCapabilities, EmbeddingResult, EngineEvent, EngineState,
    EngineStats, FinishReason, GenerationResult, ModelCapabilities, ModelClass, PoolingType,
};
pub use stream::{StreamStats, TokenBatch, TokenFrame, TokenStreamMode};

#[cfg(test)]
mod tests {
    use super::{ChatMessage, ChatRole, FinishReason, StreamStats, TokenBatch};

    #[test]
    fn shared_core_types_reexport_from_engine() {
        let message = ChatMessage::new(ChatRole::User, "hello");
        assert_eq!(message.role.as_str(), "user");

        let batch = TokenBatch {
            request_id: "request".to_string(),
            stream_id: 1,
            sequence_start: 0,
            text: "hello".to_string(),
            frame_count: 1,
            byte_count: 5,
            stats: StreamStats::default(),
        };
        assert_eq!(batch.text, "hello");
        assert_eq!(FinishReason::Stop.as_str(), "stop");
    }
}
