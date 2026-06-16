use std::time::Instant;

use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::{llama_seq_id, llama_token};

use super::{
    prefix_entry_approx_bytes, PendingPrefixSnapshot, PrefixCacheEntry, PrefixStateCache,
    PrefixStateStoreRequest,
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
    pub(crate) fn enqueue_pending_snapshot(&mut self, snapshot: PendingPrefixSnapshot) {
        if snapshot.seq_id < 0
            || snapshot.token_count == 0
            || snapshot.token_count > snapshot.prefix_tokens.len()
        {
            return;
        }

        if let Some(existing) = self
            .pending_snapshots
            .iter_mut()
            .find(|pending| same_pending_snapshot_identity(pending, &snapshot))
        {
            *existing = snapshot;
            return;
        }

        self.pending_snapshots.push_back(snapshot);
    }

    #[cfg(test)]
    pub(crate) fn pending_snapshot_count(&self) -> usize {
        self.pending_snapshots.len()
    }

    pub(crate) fn drop_pending_snapshots_for_seq(&mut self, seq_id: llama_seq_id) {
        if seq_id < 0 || self.pending_snapshots.is_empty() {
            return;
        }

        self.pending_snapshots
            .retain(|snapshot| snapshot.seq_id != seq_id);
    }

    pub(crate) fn clear_pending_snapshots(&mut self) {
        self.pending_snapshots.clear();
    }

    pub(crate) fn drain_pending_snapshots(
        &mut self,
        runtime: &NativeRuntimeHandle,
        max_to_drain: usize,
        mut sequence_is_current: impl FnMut(llama_seq_id, u64) -> bool,
        mut record_stored_snapshot: impl FnMut(usize),
    ) -> usize {
        if self.pending_snapshots.is_empty() {
            return 0;
        }

        let budget = if max_to_drain == 0 {
            self.pending_snapshots.len()
        } else {
            max_to_drain.min(self.pending_snapshots.len())
        };
        let mut drained = 0;
        while drained < budget {
            let Some(snapshot) = self.pending_snapshots.pop_front() else {
                break;
            };
            drained += 1;

            if !sequence_is_current(snapshot.seq_id, snapshot.generation) {
                continue;
            }
            let Ok(state_bytes) = runtime.state_seq(snapshot.seq_id) else {
                continue;
            };
            let token_count = snapshot.token_count;
            let stored = self.store_prefix_state(PrefixStateStoreRequest {
                state_bytes,
                seq_id: snapshot.seq_id,
                model_fingerprint: snapshot.model_fingerprint,
                snapshot_scope: &snapshot.snapshot_scope,
                tokens: &snapshot.prefix_tokens,
                token_count,
                prefix_hash: snapshot.prefix_hash,
                retention_priority: snapshot.retention_priority,
            });
            if stored {
                record_stored_snapshot(token_count);
            }
        }

        drained
    }

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
        let Ok(token_count) = i32::try_from(entry.token_count) else {
            return false;
        };
        if !runtime.set_state_seq(seq_id, &entry.state_bytes) {
            return false;
        }
        // Deferred snapshots may include KV after the recorded boundary.
        // The prefix tokens remain the identity, so trim the native sequence.
        runtime.clear_sequence(seq_id, token_count, -1)
    }
}

fn same_pending_snapshot_identity(
    left: &PendingPrefixSnapshot,
    right: &PendingPrefixSnapshot,
) -> bool {
    left.seq_id == right.seq_id
        && left.generation == right.generation
        && left.model_fingerprint == right.model_fingerprint
        && left.snapshot_scope == right.snapshot_scope
        && left.token_count == right.token_count
        && left.prefix_hash == right.prefix_hash
}
