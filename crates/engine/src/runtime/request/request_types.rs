use std::time::Instant;

use crate::engine::protocol::EmbedOptions;
use crate::runtime::config::{KvReuseMode, RequestSampling};
use crate::runtime::llama_token;
use crate::runtime::metrics::CacheSource;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/runtime/request/request_types_tests.rs"]
mod request_types_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub type GenerateRequestId = u32;
pub const NO_SAMPLED_TOKEN_ID: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerateRequestLifecycle {
    #[default]
    Pending = 0,
    Admitted,
    Running,
    Decoding,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MultimodalPayload {
    pub image_buffers: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub id: GenerateRequestId,
    pub context_key: String,
    pub original_prompt: String,
    pub grammar: String,
    pub json_schema: String,
    pub stop: Vec<String>,
    pub sampling: Option<RequestSampling>,
    pub prompt_tokens: Vec<llama_token>,
    pub multimodal: Option<MultimodalPayload>,
    /// When `Some`, this is an `embed()` request: the slot plan resolves to
    /// `TerminalAction::ReadEmbedding` and the runtime finalizes with a
    /// `ResponseOutput::Embedding` payload built from the encoder/decoder
    /// embedding read (subject to `normalize` honoring `pooling != Rank`).
    pub embed_options: Option<EmbedOptions>,
    pub max_output_tokens: i32,
    pub emit_tokens: bool,
    pub lifecycle: GenerateRequestLifecycle,
    pub enqueued_at: Option<Instant>,
    pub admitted_at: Option<Instant>,
    pub first_token_at: Option<Instant>,
    pub last_token_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub emitted_token_count: i32,
    pub itl_sum_ms: f64,
    pub itl_p99_ms: f64,
    pub e2e_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub native_sync_ms: f64,
    pub native_gpu_ms: f64,
    pub native_logic_ms: f64,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_mode: KvReuseMode,
    pub cache_source: CacheSource,
    pub cache_hits: i32,
    pub prefill_tokens: i32,
    pub first_sampled_token_id: i32,
    pub is_multimodal_turn: bool,
    pub cancel_requested: bool,
}

impl GenerateRequest {
    pub fn new(id: GenerateRequestId, context_key: impl Into<String>) -> Self {
        Self {
            id,
            context_key: context_key.into(),
            enqueued_at: Some(Instant::now()),
            ..Self::default()
        }
    }

    pub fn reset_for_queue(&mut self) {
        self.lifecycle = GenerateRequestLifecycle::Pending;
        self.admitted_at = None;
        self.first_token_at = None;
        self.last_token_at = None;
        self.completed_at = None;
        self.reset_runtime_metrics();
        self.cancel_requested = false;
    }

    fn reset_runtime_metrics(&mut self) {
        self.emitted_token_count = 0;
        self.itl_sum_ms = 0.0;
        self.itl_p99_ms = 0.0;
        self.e2e_ms = 0.0;
        self.prefill_ms = 0.0;
        self.decode_ms = 0.0;
        self.native_sync_ms = 0.0;
        self.native_gpu_ms = 0.0;
        self.native_logic_ms = 0.0;
        self.input_tokens = 0;
        self.output_tokens = 0;
        self.cache_source = CacheSource::None;
        self.cache_hits = 0;
        self.prefill_tokens = 0;
        self.first_sampled_token_id = NO_SAMPLED_TOKEN_ID;
    }
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            id: 0,
            context_key: String::new(),
            original_prompt: String::new(),
            grammar: String::new(),
            json_schema: String::new(),
            stop: Vec::new(),
            sampling: None,
            prompt_tokens: Vec::new(),
            multimodal: None,
            embed_options: None,
            max_output_tokens: 0,
            emit_tokens: false,
            lifecycle: GenerateRequestLifecycle::Pending,
            enqueued_at: None,
            admitted_at: None,
            first_token_at: None,
            last_token_at: None,
            completed_at: None,
            emitted_token_count: 0,
            itl_sum_ms: 0.0,
            itl_p99_ms: 0.0,
            e2e_ms: 0.0,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            native_sync_ms: 0.0,
            native_gpu_ms: 0.0,
            native_logic_ms: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            cache_mode: KvReuseMode::LiveSlotPrefix,
            cache_source: CacheSource::None,
            cache_hits: 0,
            prefill_tokens: 0,
            first_sampled_token_id: NO_SAMPLED_TOKEN_ID,
            is_multimodal_turn: false,
            cancel_requested: false,
        }
    }
}
