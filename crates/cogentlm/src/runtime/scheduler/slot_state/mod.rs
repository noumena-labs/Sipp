//! Per-slot state: phase, generated tokens, sampler handle, and KV mirror. One slot per active sequence.

use crate::engine::protocol::PoolingType;
use crate::native_bridge::SamplerHandle;
use crate::runtime::request::{GenerateRequest, GenerateRequestId, GenerateRequestLifecycle};
use crate::runtime::session::{CacheCandidate, KvCacheAdmission, SequenceMirror};
use crate::runtime::{llama_seq_id, llama_token};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlotPhase {
    #[default]
    Idle = 0,
    Admitted,
    Prefill,
    Decode,
    EmitBuffered,
    Completed,
    Failed,
}

/// How the slot's prompt is ingested into the runtime context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillKind {
    /// Standard `llama_decode` loop over the prompt tokens.
    #[default]
    Decode,
    /// Single `llama_encode` pass over the prompt tokens.
    Encode,
}

/// What the slot does once prompt ingest completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalAction {
    /// Enter the existing sample/decode loop and stream tokens.
    #[default]
    SampleTokens,
    /// Read the resolved embedding and finalize.
    ReadEmbedding,
}

/// Per-slot execution plan derived once at admission. The scheduler stores the
/// decision; runtime admission code owns model capability interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotExecutionPlan {
    pub prefill: PrefillKind,
    pub terminal: TerminalAction,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SamplerCacheKey {
    pub sampling_json: String,
    pub grammar: String,
    pub json_schema: String,
}

/// Pending embedding payload set by the embedding-read path. `finalize_completed_slots`
/// drains this into `ResponseOutput::Embedding` when the slot terminates.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotEmbeddingOutput {
    pub values: Vec<f32>,
    pub pooling: PoolingType,
    pub normalized: bool,
}

#[derive(Debug)]
pub struct SlotState {
    pub slot_id: usize,
    pub seq_id: llama_seq_id,
    pub phase: SlotPhase,
    pub plan: SlotExecutionPlan,
    pub request_id: GenerateRequestId,
    pub request: Option<GenerateRequest>,
    pub lease_generation: u64,
    pub cache_candidate: CacheCandidate,
    pub mirror: SequenceMirror,
    pub prefill_cursor: usize,
    pub decode_step_count: usize,
    pub batch_participation_count: usize,
    pub generated_tokens: Vec<llama_token>,
    pub output_text: String,
    pub buffered_output_text: String,
    pub pending_emission_text: String,
    pub pending_utf8_bytes: Vec<u8>,
    pub terminal_error_message: String,
    pub sampler_prompt_seeded: bool,
    pub sampler_key: Option<SamplerCacheKey>,
    pub(crate) backend_sampler_attached: bool,
    pub(crate) sampler: Option<SamplerHandle>,
    pub embedding_output: Option<SlotEmbeddingOutput>,
}

impl SlotState {
    #[cfg(test)]
    pub fn new(slot_id: usize) -> Self {
        let mut slot = Self::default();
        slot.slot_id = slot_id;
        slot
    }

    pub fn reset_to_idle(&mut self) {
        debug_assert!(
            !self.backend_sampler_attached,
            "backend sampler must be detached before slot reset"
        );
        self.free_sampler();
        self.sampler_key = None;
        self.phase = SlotPhase::Idle;
        self.seq_id = -1;
        self.request_id = 0;
        self.request = None;
        self.lease_generation = 0;
        self.cache_candidate = CacheCandidate::None;
        self.backend_sampler_attached = false;
        self.mirror = SequenceMirror::default();
        self.clear_request_progress();
    }

    pub fn attach_request(&mut self, request: GenerateRequest, admission: KvCacheAdmission) {
        debug_assert!(
            !self.backend_sampler_attached,
            "backend sampler must be detached before replacing slot request"
        );
        self.free_sampler();
        self.sampler_key = None;
        self.request_id = request.id;
        self.request = Some(request);
        self.seq_id = admission.seq_id;
        self.lease_generation = admission.generation;
        self.cache_candidate = admission.candidate;
        self.mirror = admission.mirror;
        self.phase = SlotPhase::Admitted;
        self.backend_sampler_attached = false;
        self.clear_request_progress();
    }

    fn clear_request_progress(&mut self) {
        self.prefill_cursor = 0;
        self.decode_step_count = 0;
        self.batch_participation_count = 0;
        self.generated_tokens.clear();
        self.output_text.clear();
        self.buffered_output_text.clear();
        self.pending_emission_text.clear();
        self.pending_utf8_bytes.clear();
        self.terminal_error_message.clear();
        self.sampler_prompt_seeded = false;
        self.embedding_output = None;
    }

    pub fn request(&self) -> Option<&GenerateRequest> {
        self.request.as_ref()
    }

    pub fn request_mut(&mut self) -> Option<&mut GenerateRequest> {
        self.request.as_mut()
    }

    pub fn fail(&mut self, message: impl Into<String>) {
        self.terminal_error_message = message.into();
        self.phase = SlotPhase::Failed;
        if let Some(request) = self.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Failed;
        }
    }

    pub fn cancel(&mut self, message: impl Into<String>) {
        self.terminal_error_message = message.into();
        self.phase = SlotPhase::Failed;
        if let Some(request) = self.request_mut() {
            request.lifecycle = GenerateRequestLifecycle::Cancelled;
        }
    }

    pub(crate) fn set_sampler(&mut self, sampler: SamplerHandle) {
        debug_assert!(
            !self.backend_sampler_attached,
            "backend sampler must be detached before replacing sampler"
        );
        self.free_sampler();
        self.sampler_key = None;
        self.sampler = Some(sampler);
        self.sampler_prompt_seeded = false;
        self.backend_sampler_attached = false;
    }

    pub(crate) fn take_sampler(&mut self) -> Option<SamplerHandle> {
        debug_assert!(
            !self.backend_sampler_attached,
            "backend sampler must be detached before taking sampler"
        );
        self.sampler_key = None;
        self.sampler.take()
    }

    fn free_sampler(&mut self) {
        self.sampler = None;
    }
}

impl Default for SlotState {
    fn default() -> Self {
        Self {
            slot_id: 0,
            seq_id: -1,
            phase: SlotPhase::Idle,
            plan: SlotExecutionPlan::default(),
            request_id: 0,
            request: None,
            lease_generation: 0,
            cache_candidate: CacheCandidate::None,
            mirror: SequenceMirror::default(),
            prefill_cursor: 0,
            decode_step_count: 0,
            batch_participation_count: 0,
            generated_tokens: Vec::new(),
            output_text: String::new(),
            buffered_output_text: String::new(),
            pending_emission_text: String::new(),
            pending_utf8_bytes: Vec::new(),
            terminal_error_message: String::new(),
            sampler_prompt_seeded: false,
            backend_sampler_attached: false,
            sampler: None,
            sampler_key: None,
            embedding_output: None,
        }
    }
}

impl Drop for SlotState {
    fn drop(&mut self) {
        self.free_sampler();
    }
}

#[cfg(test)]
#[path = "../../../tests/runtime/scheduler/slot_state_tests.rs"]
mod slot_state_tests;
