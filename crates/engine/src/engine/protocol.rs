use serde::{Deserialize, Serialize};

pub use cogentlm_core::FinishReason;

use crate::runtime::config::{KvReuseMode, ResolvedRuntimeLimits};
pub use crate::runtime::metrics::CacheSource;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/engine/protocol_tests.rs"]
mod protocol_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Current lifecycle state of an engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    Idle,
    Loading,
    Ready,
    Running,
    Error,
    Closed,
}

impl EngineStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Error => "error",
            Self::Closed => "closed",
        }
    }
}

/// Current lifecycle state of a live request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Queued,
    Prefill,
    Decode,
    Completed,
    Failed,
    Cancelled,
}

impl RequestStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendDevice {
    pub id: Option<String>,
    pub name: String,
    pub device_type: String,
    pub memory_total_bytes: Option<u64>,
    pub memory_free_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BackendInfo {
    pub selected: String,
    pub available: Vec<String>,
    pub devices: Vec<BackendDevice>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelState {
    pub id: String,
    pub name: String,
    pub capabilities: ModelCapabilities,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RequestState {
    pub id: String,
    pub status: RequestStatus,
    pub input_tokens: i32,
    pub output_tokens: i32,
}

/// Numeric engine measurements. Exporters can map these fields to metrics.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EngineStats {
    pub requests_running: i32,
    pub requests_queued: i32,
    pub requests_completed: i64,
    pub requests_failed: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_mode: KvReuseMode,
    pub cache_source: CacheSource,
    pub cache_hits: i64,
    pub prefill_tokens: i64,
    pub ttft_ms: Option<f64>,
    pub inter_token_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    /// End-to-end output-token throughput, including queue/admit/prefill/decode/finalization.
    pub e2e_tokens_per_second: Option<f64>,
    /// Decode-only output-token throughput, excluding prefill/TTFT.
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub backend_ms: f64,
    pub sync_ms: f64,
    pub engine_overhead_ms: f64,
}

/// Point-in-time engine state. This is the canonical public state view.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineState {
    pub status: EngineStatus,
    pub model: Option<ModelState>,
    pub backend: BackendInfo,
    pub runtime: Option<ResolvedRuntimeLimits>,
    pub requests: Vec<RequestState>,
    pub stats: EngineStats,
    pub updated_at_unix_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RequestStats {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_mode: KvReuseMode,
    pub cache_source: CacheSource,
    pub cache_hits: i32,
    pub prefill_tokens: i32,
    pub ttft_ms: Option<f64>,
    pub inter_token_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    /// End-to-end output-token throughput, including queue/admit/prefill/decode/finalization.
    pub e2e_tokens_per_second: Option<f64>,
    /// Decode-only output-token throughput, excluding prefill/TTFT.
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

/// Final text output and stats for a `query()` / `chat()` request.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    pub id: String,
    pub text: String,
    pub finish_reason: FinishReason,
    pub stats: RequestStats,
}

/// Pooling strategy for embedding outputs. Mirrors `llama_pooling_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingType {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

impl PoolingType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::None => "none",
            Self::Mean => "mean",
            Self::Cls => "cls",
            Self::Last => "last",
            Self::Rank => "rank",
        }
    }

    pub fn from_name(value: &str) -> Option<Self> {
        match value {
            "unspecified" => Some(Self::Unspecified),
            "none" => Some(Self::None),
            "mean" => Some(Self::Mean),
            "cls" => Some(Self::Cls),
            "last" => Some(Self::Last),
            "rank" => Some(Self::Rank),
            _ => None,
        }
    }

    pub const fn from_llama_value(value: i32) -> Option<Self> {
        match value {
            -1 => Some(Self::Unspecified),
            0 => Some(Self::None),
            1 => Some(Self::Mean),
            2 => Some(Self::Cls),
            3 => Some(Self::Last),
            4 => Some(Self::Rank),
            _ => None,
        }
    }

    pub const fn is_explicit(self) -> bool {
        !matches!(self, Self::Unspecified)
    }
}

/// llama.cpp model-shape classification. Determines which inference path
/// `query()` and `embed()` take: see `cogentlm_encoder.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelClass {
    /// `has_encoder = false`, `has_decoder = true`.
    DecoderOnly,
    /// `has_encoder = true`, `has_decoder = true`.
    EncoderDecoder,
    /// `has_encoder = true`, `has_decoder = false` (BERT, e5, bge, jina, reranker).
    EncoderOnly,
}

impl ModelClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DecoderOnly => "decoder_only",
            Self::EncoderDecoder => "encoder_decoder",
            Self::EncoderOnly => "encoder_only",
        }
    }

    /// Classify by `general.architecture` (already lowercased by the
    /// inspection layer).
    pub fn from_architecture(arch: &str) -> Self {
        match arch {
            "bert" | "nomic-bert" | "nomic-bert-moe" | "jina-bert-v2" | "jina-bert-v3"
            | "modern-bert" | "neo-bert" | "t5encoder" => Self::EncoderOnly,
            "t5" | "bart" => Self::EncoderDecoder,
            _ => Self::DecoderOnly,
        }
    }
}

/// Embedding output capability for the currently-loaded model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingCapabilities {
    pub dimensions: i32,
    pub pooling: PoolingType,
}

/// Public capability snapshot for the currently-loaded model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCapabilities {
    pub model_class: ModelClass,
    pub supports_text_generation: bool,
    pub supports_embeddings: bool,
    pub has_chat_template: bool,
    pub embedding: Option<EmbeddingCapabilities>,
}

/// Per-call knobs for `embed()`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedOptions {
    /// L2-normalize the returned vector. Ignored for `pooling = Rank`.
    pub normalize: bool,
    pub context_key: Option<String>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            normalize: true,
            context_key: None,
        }
    }
}

/// Public `embed()` request.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedRequest {
    pub input: String,
    pub options: EmbedOptions,
}

/// Final embedding output and stats for an `embed()` request.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResult {
    pub id: String,
    pub values: Vec<f32>,
    pub pooling: PoolingType,
    pub normalized: bool,
    pub stats: RequestStats,
}

/// Runtime event emitted by the engine actor.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    State(Box<EngineState>),
    LoadProgress {
        loaded_bytes: u64,
        total_bytes: Option<u64>,
        asset_name: Option<String>,
    },
    RequestStarted {
        request_id: String,
        stream_id: u32,
    },
    RequestCompleted {
        request_id: String,
    },
    RequestFailed {
        request_id: String,
        error: String,
    },
    Closed,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            status: EngineStatus::Idle,
            model: None,
            backend: BackendInfo::default(),
            runtime: None,
            requests: Vec::new(),
            stats: EngineStats::default(),
            updated_at_unix_ms: 0,
        }
    }
}
