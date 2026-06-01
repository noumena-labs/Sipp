//! Reusable llama_batch builder. Owns the input arrays so the runtime can fill
//! a fresh batch each tick without reallocating.

use crate::error::Result;
use crate::native_bridge::NativeBatchHandle;
use crate::runtime::{llama_seq_id, llama_token};

pub struct LlamaBatchBuilder {
    batch: NativeBatchHandle,
}

impl Default for LlamaBatchBuilder {
    fn default() -> Self {
        Self {
            batch: NativeBatchHandle::new(),
        }
    }
}

impl LlamaBatchBuilder {
    pub fn ensure_capacity(&mut self, max_tokens: i32, max_sequences: i32) -> Result<()> {
        let max_tokens = max_tokens.max(1);
        let max_sequences = max_sequences.max(1);

        self.batch.ensure_capacity(max_tokens, max_sequences)
    }

    pub fn reset(&mut self) {
        self.batch.reset();
    }

    /// Called once per token in every scheduler tick. With codegen-units=1 +
    /// LTO this gets inlined into `run_policy_batch_tick_locked` and the
    /// per-token cost collapses to a few pointer writes.
    #[inline]
    pub fn add_token(
        &mut self,
        token: llama_token,
        position: i32,
        seq_id: llama_seq_id,
        request_logits: bool,
    ) -> bool {
        self.batch
            .add_token(token, position, seq_id, request_logits)
    }

    pub(crate) fn batch(&self) -> &NativeBatchHandle {
        &self.batch
    }

    pub(crate) fn n_tokens(&self) -> i32 {
        self.batch.n_tokens()
    }

    #[cfg(test)]
    fn token(&self, index: i32) -> i32 {
        self.batch.token(index)
    }

    #[cfg(test)]
    fn pos(&self, index: i32) -> i32 {
        self.batch.pos(index)
    }

    #[cfg(test)]
    fn seq_id(&self, index: i32) -> i32 {
        self.batch.seq_id(index)
    }

    #[cfg(test)]
    fn logits(&self, index: i32) -> bool {
        self.batch.logits(index)
    }
}

#[cfg(test)]
#[path = "../../../tests/runtime/llama/llama_batch_builder_tests.rs"]
mod llama_batch_builder_tests;
