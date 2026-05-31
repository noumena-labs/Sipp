//! Per-slot state: phase, generated tokens, sampler handle, and KV mirror. One slot per active sequence.

use std::ptr::NonNull;

use cogentlm_sys as ffi;

use crate::engine::protocol::PoolingType;
use crate::runtime::request::{GenerateRequest, GenerateRequestId, GenerateRequestLifecycle};
use crate::runtime::session::SequenceState;
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
    pub session: Option<SequenceState>,
    pub mirror: SequenceState,
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
    pub sampler: Option<NonNull<ffi::cogent_common_sampler>>,
    pub embedding_output: Option<SlotEmbeddingOutput>,
}

impl SlotState {
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
        self.session = None;
        self.backend_sampler_attached = false;
        self.mirror = SequenceState::default();
        self.clear_request_progress();
    }

    pub fn attach_request(&mut self, request: GenerateRequest, mut session: SequenceState) {
        debug_assert!(
            !self.backend_sampler_attached,
            "backend sampler must be detached before replacing slot request"
        );
        self.free_sampler();
        self.sampler_key = None;
        self.request_id = request.id;
        self.request = Some(request);
        self.mirror.current_kv_tokens = std::mem::take(&mut session.current_kv_tokens);
        self.mirror.n_past = session.n_past;
        self.mirror.hardware_id = session.hardware_id;
        self.session = Some(session);
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

    pub fn set_sampler(&mut self, sampler: *mut ffi::cogent_common_sampler) {
        debug_assert!(
            !self.backend_sampler_attached,
            "backend sampler must be detached before replacing sampler"
        );
        self.free_sampler();
        self.sampler_key = None;
        self.sampler = NonNull::new(sampler);
        self.sampler_prompt_seeded = false;
        self.backend_sampler_attached = false;
    }

    pub fn take_sampler(&mut self) -> Option<NonNull<ffi::cogent_common_sampler>> {
        debug_assert!(
            !self.backend_sampler_attached,
            "backend sampler must be detached before taking sampler"
        );
        self.sampler_key = None;
        self.sampler.take()
    }

    fn free_sampler(&mut self) {
        if let Some(sampler) = self.sampler.take() {
            // SAFETY: SlotState owns sampler pointers installed through
            // set_sampler unless take_sampler transfers ownership to the
            // runtime sampler pool. NonNull guarantees a non-null pointer, and
            // this path takes the Option so the sampler is freed at most once.
            unsafe {
                ffi::cogent_common_sampler_free(sampler.as_ptr());
            }
        }
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
            session: None,
            mirror: SequenceState::default(),
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
mod tests {
    mod slot_state_tests;
}
