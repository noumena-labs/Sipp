use crate::runtime::config::ResolvedRuntimeLimits;

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

/// Why a request stopped producing tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

impl FinishReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
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
    pub cache_hits: i64,
    pub prefill_tokens: i64,
    pub ttft_ms: Option<f64>,
    pub inter_token_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    /// End-to-end output-token throughput, including queue/admit/prefill/decode/finalization.
    pub tokens_per_second: Option<f64>,
    /// Decode-only output-token throughput, excluding prefill/TTFT.
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_tokens_per_second: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub backend_ms: f64,
    pub sync_ms: f64,
    pub engine_overhead_ms: f64,
    pub debug_metrics_scheduler_ticks: i64,
    pub debug_metrics_decode_ticks: i64,
    pub debug_metrics_prefill_ticks: i64,
    pub debug_metrics_backend_sampler_attach_attempts: i64,
    pub debug_metrics_backend_sampler_attach_failures: i64,
    pub debug_metrics_admit_ms: f64,
    pub debug_metrics_normalize_ms: f64,
    pub debug_metrics_backend_sampler_attach_ms: f64,
    pub debug_metrics_select_slots_ms: f64,
    pub debug_metrics_plan_ms: f64,
    pub debug_metrics_batch_build_ms: f64,
    pub debug_metrics_llama_decode_ms: f64,
    pub debug_metrics_llama_sync_ms: f64,
    pub debug_metrics_apply_bookkeeping_ms: f64,
    pub debug_metrics_apply_decode_results_ms: f64,
    pub debug_metrics_sample_ms: f64,
    pub debug_metrics_token_piece_ms: f64,
    pub debug_metrics_emit_ms: f64,
    pub debug_metrics_prefix_queue_ms: f64,
    pub debug_metrics_finalize_ms: f64,
    pub debug_metrics_commit_observability_ms: f64,
    pub debug_metrics_post_decode_ms: f64,
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
    pub cache_hits: i32,
    pub ttft_ms: Option<f64>,
    pub inter_token_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    /// End-to-end output-token throughput, including queue/admit/prefill/decode/finalization.
    pub tokens_per_second: Option<f64>,
    /// Decode-only output-token throughput, excluding prefill/TTFT.
    pub decode_tokens_per_second: Option<f64>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub debug_metrics_scheduler_ticks: i32,
    pub debug_metrics_decode_ticks: i32,
    pub debug_metrics_prefill_ticks: i32,
    pub debug_metrics_backend_sampler_attach_attempts: i32,
    pub debug_metrics_backend_sampler_attach_failures: i32,
    pub debug_metrics_admit_ms: f64,
    pub debug_metrics_normalize_ms: f64,
    pub debug_metrics_backend_sampler_attach_ms: f64,
    pub debug_metrics_select_slots_ms: f64,
    pub debug_metrics_plan_ms: f64,
    pub debug_metrics_batch_build_ms: f64,
    pub debug_metrics_llama_decode_ms: f64,
    pub debug_metrics_llama_sync_ms: f64,
    pub debug_metrics_apply_bookkeeping_ms: f64,
    pub debug_metrics_apply_decode_results_ms: f64,
    pub debug_metrics_sample_ms: f64,
    pub debug_metrics_token_piece_ms: f64,
    pub debug_metrics_emit_ms: f64,
    pub debug_metrics_prefix_queue_ms: f64,
    pub debug_metrics_finalize_ms: f64,
    pub debug_metrics_commit_observability_ms: f64,
    pub debug_metrics_post_decode_ms: f64,
}

/// Final output and final stats for one request.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestResult {
    pub id: String,
    pub text: String,
    pub finish_reason: FinishReason,
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
        result: Box<RequestResult>,
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
