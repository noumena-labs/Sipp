//! Reusable llama_batch builder. Owns the input arrays so the runtime can fill a fresh batch each tick without reallocating.

use cogentlm_sys as ffi;

use crate::error::{Error, Result};
use crate::runtime::{llama_seq_id, llama_token};

pub struct LlamaBatchBuilder {
    batch: ffi::llama_batch,
    capacity_tokens: i32,
    capacity_sequences: i32,
    is_allocated: bool,
}

impl Default for LlamaBatchBuilder {
    fn default() -> Self {
        Self {
            batch: empty_batch(),
            capacity_tokens: 0,
            capacity_sequences: 0,
            is_allocated: false,
        }
    }
}

impl LlamaBatchBuilder {
    pub fn ensure_capacity(&mut self, max_tokens: i32, max_sequences: i32) -> Result<()> {
        let max_tokens = max_tokens.max(1);
        let max_sequences = max_sequences.max(1);

        if self.is_allocated
            && self.capacity_tokens >= max_tokens
            && self.capacity_sequences >= max_sequences
        {
            self.reset();
            return Ok(());
        }

        self.free();
        // SAFETY: llama_batch_init is an FFI constructor. max_tokens and
        // max_sequences are clamped to positive values and ownership of the
        // returned allocation is stored in self until freed by self.free/Drop.
        self.batch = unsafe { ffi::llama_batch_init(max_tokens, 0, max_sequences) };
        if self.batch.token.is_null()
            || self.batch.pos.is_null()
            || self.batch.n_seq_id.is_null()
            || self.batch.seq_id.is_null()
            || self.batch.logits.is_null()
        {
            // SAFETY: self.batch is the allocation just returned by
            // llama_batch_init. std::ptr::read moves the C-owned batch value to
            // the FFI destructor without running Rust drop glue.
            unsafe {
                ffi::llama_batch_free(std::ptr::read(&self.batch));
            }
            self.clear_storage_fields();
            return Err(Error::NullPointer("llama_batch_init"));
        }

        self.capacity_tokens = max_tokens;
        self.capacity_sequences = max_sequences;
        self.is_allocated = true;
        self.reset();
        Ok(())
    }

    pub fn reset(&mut self) {
        if self.is_allocated {
            self.batch.n_tokens = 0;
        }
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
        if !self.is_allocated
            || self.batch.n_tokens < 0
            || self.batch.n_tokens >= self.capacity_tokens
        {
            return false;
        }

        let Ok(index) = usize::try_from(self.batch.n_tokens) else {
            return false;
        };
        let Some(next_token_count) = self.batch.n_tokens.checked_add(1) else {
            return false;
        };
        // SAFETY: ensure_capacity verified all batch arrays are non-null and
        // capacity_tokens bounds n_tokens. llama_batch_init allocates at least
        // capacity_tokens entries for token/pos/n_seq_id/logits and at least
        // capacity_sequences sequence-id cells per token; this builder writes
        // one sequence id per token.
        unsafe {
            let seq_ids = *self.batch.seq_id.add(index);
            if seq_ids.is_null() {
                return false;
            }
            *self.batch.token.add(index) = token;
            *self.batch.pos.add(index) = position;
            *self.batch.n_seq_id.add(index) = 1;
            *seq_ids = seq_id;
            *self.batch.logits.add(index) = i8::from(request_logits);
        }
        self.batch.n_tokens = next_token_count;
        true
    }

    pub fn is_allocated(&self) -> bool {
        self.is_allocated
    }

    pub fn capacity_tokens(&self) -> i32 {
        self.capacity_tokens
    }

    pub fn capacity_sequences(&self) -> i32 {
        self.capacity_sequences
    }

    pub fn raw(&self) -> &ffi::llama_batch {
        &self.batch
    }

    pub fn raw_mut(&mut self) -> &mut ffi::llama_batch {
        &mut self.batch
    }

    fn free(&mut self) {
        if !self.is_allocated {
            return;
        }

        // SAFETY: self.batch owns an allocation returned by llama_batch_init
        // and has not been freed while is_allocated is true. ptr::read moves
        // the FFI value into the matching destructor exactly once.
        unsafe {
            ffi::llama_batch_free(std::ptr::read(&self.batch));
        }
        self.clear_storage_fields();
    }

    fn clear_storage_fields(&mut self) {
        self.batch = empty_batch();
        self.capacity_tokens = 0;
        self.capacity_sequences = 0;
        self.is_allocated = false;
    }
}

impl Drop for LlamaBatchBuilder {
    fn drop(&mut self) {
        self.free();
    }
}

fn empty_batch() -> ffi::llama_batch {
    ffi::llama_batch {
        n_tokens: 0,
        token: std::ptr::null_mut(),
        embd: std::ptr::null_mut(),
        pos: std::ptr::null_mut(),
        n_seq_id: std::ptr::null_mut(),
        seq_id: std::ptr::null_mut(),
        logits: std::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests;
