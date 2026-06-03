use std::time::Instant;

use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::{llama_seq_id, llama_token};

use super::{
    prefix_entry_approx_bytes, PrefixCacheEntry, PrefixStateCache, PrefixStateStoreRequest,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../../tests/runtime/session/prefix_state_cache/state_io_tests.rs"]
mod state_io_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

impl PrefixStateCache {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_prefix_state(
        &mut self,
        runtime: &NativeRuntimeHandle,
        seq_id: llama_seq_id,
        model_fingerprint: u64,
        snapshot_scope: &str,
        tokens: &[llama_token],
        token_count: usize,
        prefix_hash: u64,
        retention_priority: u64,
    ) -> bool {
        let Ok(state_bytes) = runtime.state_seq(seq_id) else {
            return false;
        };
        self.store_prefix_state(PrefixStateStoreRequest {
            state_bytes,
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
        if request.seq_id < 0
            || request.token_count == 0
            || request.token_count > request.tokens.len()
            || request.state_bytes.is_empty()
        {
            return false;
        }

        let Some(approx_bytes) =
            prefix_entry_approx_bytes(request.state_bytes.len(), request.token_count)
        else {
            return false;
        };

        self.insert_or_update_entry(PrefixCacheEntry {
            model_fingerprint: request.model_fingerprint,
            snapshot_scope: request.snapshot_scope.to_string(),
            token_count: request.token_count,
            prefix_hash: request.prefix_hash,
            retention_priority: request.retention_priority,
            hit_count: 0,
            approx_bytes,
            prefix_tokens: request.tokens[..request.token_count].to_vec(),
            state_bytes: request.state_bytes,
            last_used: Instant::now(),
        });
        true
    }

    pub(crate) fn restore_prefix_state(
        &self,
        runtime: &mut NativeRuntimeHandle,
        seq_id: llama_seq_id,
        entry: &PrefixCacheEntry,
    ) -> bool {
        if seq_id < 0 || entry.state_bytes.is_empty() {
            return false;
        }
        runtime.set_state_seq(seq_id, &entry.state_bytes)
    }
}
