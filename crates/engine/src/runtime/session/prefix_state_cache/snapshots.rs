use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::llama_seq_id;

use super::{PendingPrefixSnapshot, PrefixStateCache, PrefixStateStoreRequest};

impl PrefixStateCache {
    pub fn enqueue_pending_snapshot(&mut self, snapshot: PendingPrefixSnapshot) {
        if snapshot.seq_id < 0
            || snapshot.token_count == 0
            || snapshot.prefix_tokens.len() < snapshot.token_count
        {
            return;
        }

        if let Some(existing) = self.pending_snapshots.iter_mut().find(|pending| {
            pending.seq_id == snapshot.seq_id
                && pending.context_key == snapshot.context_key
                && pending.model_fingerprint == snapshot.model_fingerprint
                && pending.token_count == snapshot.token_count
        }) {
            *existing = snapshot;
            return;
        }

        self.pending_snapshots.push_back(snapshot);
    }

    pub(crate) fn drain_pending_snapshots(
        &mut self,
        runtime: &NativeRuntimeHandle,
        max_to_drain: usize,
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
            let Some(pending) = self.pending_snapshots.pop_front() else {
                break;
            };
            self.store_pending_snapshot(runtime, pending);
            drained += 1;
        }
        drained
    }

    pub(crate) fn drain_best_pending_snapshot_for_seq(
        &mut self,
        runtime: &NativeRuntimeHandle,
        seq_id: llama_seq_id,
    ) -> usize {
        if seq_id < 0 || self.pending_snapshots.is_empty() {
            return 0;
        }

        let best_index = self
            .pending_snapshots
            .iter()
            .enumerate()
            .filter(|(_, pending)| pending.seq_id == seq_id)
            .max_by_key(|(_, pending)| pending.token_count)
            .map(|(index, _)| index);

        let drained = best_index
            .and_then(|index| self.pending_snapshots.remove(index))
            .map(|pending| {
                self.store_pending_snapshot(runtime, pending);
                1
            })
            .unwrap_or(0);

        self.pending_snapshots
            .retain(|snapshot| snapshot.seq_id != seq_id);
        drained
    }

    pub fn drop_pending_snapshots_for_seq(&mut self, seq_id: llama_seq_id) {
        if seq_id < 0 {
            return;
        }
        self.pending_snapshots
            .retain(|snapshot| snapshot.seq_id != seq_id);
    }

    fn store_pending_snapshot(
        &mut self,
        runtime: &NativeRuntimeHandle,
        pending: PendingPrefixSnapshot,
    ) {
        let Ok(state_bytes) = runtime.state_seq(pending.seq_id) else {
            return;
        };
        self.store_prefix_state(PrefixStateStoreRequest {
            state_bytes,
            seq_id: pending.seq_id,
            model_fingerprint: pending.model_fingerprint,
            context_key: &pending.context_key,
            tokens: &pending.prefix_tokens,
            token_count: pending.token_count,
            prefix_hash: pending.prefix_hash,
            retention_priority: pending.retention_priority,
        });
    }
}

#[cfg(test)]
mod tests {
    mod snapshots_tests;
}
