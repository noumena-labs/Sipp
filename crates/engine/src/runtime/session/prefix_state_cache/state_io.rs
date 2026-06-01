use std::ffi::c_void;
use std::time::Instant;

use cogentlm_sys as ffi;

use crate::runtime::{llama_seq_id, llama_token};

use super::{
    prefix_entry_approx_bytes, PrefixCacheEntry, PrefixStateCache, PrefixStateStoreRequest,
};

impl PrefixStateCache {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_prefix_state(
        &mut self,
        context: *mut ffi::llama_context,
        seq_id: llama_seq_id,
        model_fingerprint: u64,
        snapshot_scope: &str,
        tokens: &[llama_token],
        token_count: usize,
        prefix_hash: u64,
        retention_priority: u64,
    ) -> bool {
        self.store_prefix_state(PrefixStateStoreRequest {
            context,
            seq_id,
            model_fingerprint,
            snapshot_scope,
            tokens,
            token_count,
            prefix_hash,
            retention_priority,
        })
    }

    pub(super) fn store_prefix_state(&mut self, request: PrefixStateStoreRequest<'_>) -> bool {
        if request.context.is_null()
            || request.seq_id < 0
            || request.token_count == 0
            || request.token_count > request.tokens.len()
        {
            return false;
        }

        let mut data_ptr: *mut u8 = std::ptr::null_mut();
        let mut prefix_state_size = 0_usize;
        // SAFETY: `request.context` is checked non-null above and `seq_id`
        // is non-negative. The shim allocates `data_ptr` and reports its
        // byte length; this function frees the buffer exactly once below.
        let ok = unsafe {
            ffi::cogent_llama_state_seq_get_data_ext_alloc(
                request.context,
                request.seq_id,
                ffi::LLAMA_STATE_SEQ_FLAGS_NONE,
                &mut data_ptr,
                &mut prefix_state_size,
            )
        };
        if !ok || data_ptr.is_null() || prefix_state_size == 0 {
            return false;
        }

        let Some(approx_bytes) = prefix_entry_approx_bytes(prefix_state_size, request.token_count)
        else {
            // SAFETY: `data_ptr` was allocated by the shim call above and has
            // not been freed yet on this path.
            unsafe {
                ffi::cogent_free_buffer(data_ptr.cast::<c_void>());
            }
            return false;
        };
        // SAFETY: The shim returned a non-null buffer with
        // `prefix_state_size` bytes. We copy it into Rust-owned storage before
        // freeing the C allocation.
        let state_bytes =
            unsafe { std::slice::from_raw_parts(data_ptr, prefix_state_size) }.to_vec();
        // SAFETY: `data_ptr` was allocated by the shim call above and the
        // slice copy is complete, so the native buffer can be released.
        unsafe {
            ffi::cogent_free_buffer(data_ptr.cast::<c_void>());
        }

        self.insert_or_update_entry(PrefixCacheEntry {
            model_fingerprint: request.model_fingerprint,
            snapshot_scope: request.snapshot_scope.to_string(),
            token_count: request.token_count,
            prefix_hash: request.prefix_hash,
            retention_priority: request.retention_priority,
            hit_count: 0,
            approx_bytes,
            prefix_tokens: request.tokens[..request.token_count].to_vec(),
            state_bytes,
            last_used: Instant::now(),
        });
        true
    }

    pub(crate) fn restore_prefix_state(
        &self,
        context: *mut ffi::llama_context,
        seq_id: llama_seq_id,
        entry: &PrefixCacheEntry,
    ) -> bool {
        if context.is_null() || seq_id < 0 || entry.state_bytes.is_empty() {
            return false;
        }
        // SAFETY: `context` is checked non-null and `state_bytes` is
        // immutable for the duration of the call. The shim copies the bytes
        // into llama.cpp sequence state for the provided non-negative seq id.
        unsafe {
            ffi::cogent_llama_state_seq_set_data_ext(
                context,
                seq_id,
                ffi::LLAMA_STATE_SEQ_FLAGS_NONE,
                entry.state_bytes.as_ptr(),
                entry.state_bytes.len(),
            )
        }
    }
}
