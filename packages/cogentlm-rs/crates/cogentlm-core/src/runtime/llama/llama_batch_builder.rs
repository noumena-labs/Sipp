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
            && self.capacity_tokens == max_tokens
            && self.capacity_sequences == max_sequences
        {
            self.reset();
            return Ok(());
        }

        self.free();
        self.batch = unsafe { ffi::llama_batch_init(max_tokens, 0, max_sequences) };
        if self.batch.token.is_null()
            || self.batch.pos.is_null()
            || self.batch.n_seq_id.is_null()
            || self.batch.seq_id.is_null()
            || self.batch.logits.is_null()
        {
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
        if !self.is_allocated || self.batch.n_tokens >= self.capacity_tokens {
            return false;
        }

        let index = self.batch.n_tokens as isize;
        unsafe {
            let seq_ids = *self.batch.seq_id.offset(index);
            if seq_ids.is_null() {
                return false;
            }
            *self.batch.token.offset(index) = token;
            *self.batch.pos.offset(index) = position;
            *self.batch.n_seq_id.offset(index) = 1;
            *seq_ids = seq_id;
            *self.batch.logits.offset(index) = i8::from(request_logits);
        }
        self.batch.n_tokens += 1;
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
mod tests {
    use super::*;

    #[test]
    fn ensure_capacity_allocates_and_reuses_matching_batch() {
        let mut builder = LlamaBatchBuilder::default();
        builder.ensure_capacity(2, 1).expect("allocate");
        assert!(builder.is_allocated());
        assert_eq!(builder.capacity_tokens(), 2);
        assert!(builder.add_token(10, 0, 0, false));
        assert_eq!(builder.raw().n_tokens, 1);

        builder.ensure_capacity(2, 1).expect("reuse");
        assert_eq!(builder.raw().n_tokens, 0);
        assert_eq!(builder.capacity_tokens(), 2);
    }

    #[test]
    fn add_token_populates_batch_arrays_and_clamps_capacity() {
        let mut builder = LlamaBatchBuilder::default();
        builder.ensure_capacity(1, 1).expect("allocate");

        assert!(builder.add_token(42, 7, 3, true));
        assert!(!builder.add_token(43, 8, 3, false));
        assert_eq!(builder.raw().n_tokens, 1);
        unsafe {
            assert_eq!(*builder.raw().token, 42);
            assert_eq!(*builder.raw().pos, 7);
            assert_eq!(*builder.raw().n_seq_id, 1);
            assert_eq!(**builder.raw().seq_id, 3);
            assert_eq!(*builder.raw().logits, 1);
        }
    }
}
