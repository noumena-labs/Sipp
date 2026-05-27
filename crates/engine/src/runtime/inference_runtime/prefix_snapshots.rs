use crate::runtime::numeric::saturating_usize_to_u64;
use crate::runtime::scheduler::{BatchContributionKind, SlotPhase};
use crate::runtime::session::PendingPrefixSnapshot;

use super::{unique_slot_first_use, InferenceRuntime};

impl InferenceRuntime {
    pub(super) fn queue_prefix_snapshots(
        &mut self,
        plan: &crate::runtime::scheduler::SharedBatchPlan,
    ) {
        if !self.scratch_decode_ready_slots.is_empty() {
            return;
        }

        let mut seen_slots: u64 = 0;
        for contribution in &plan.contributions {
            if contribution.kind != BatchContributionKind::Prefill
                || !unique_slot_first_use(&mut seen_slots, contribution.slot_index)
            {
                continue;
            }
            let Some(slot) = self.slot_scheduler.slots.get(contribution.slot_index) else {
                continue;
            };
            let Some(request) = slot.request() else {
                continue;
            };
            let token_count = slot.mirror.current_kv_tokens.len();
            if !self
                .prefix_cache_policy
                .should_store_boundary(token_count, request.prompt_tokens.len())
            {
                continue;
            }
            self.prefix_state_cache
                .enqueue_pending_snapshot(PendingPrefixSnapshot {
                    model_fingerprint: self.model_fingerprint,
                    context_key: request.context_key.clone(),
                    seq_id: slot.seq_id,
                    token_count,
                    prefix_hash: self
                        .prefix_cache_policy
                        .hash_prefix(&slot.mirror.current_kv_tokens, token_count),
                    retention_priority: saturating_usize_to_u64(token_count),
                    prefix_tokens: slot.mirror.current_kv_tokens[..token_count].to_vec(),
                });
            self.prefix_cache_policy.record_store(token_count);
        }
    }

    pub(super) fn resolve_terminal_prefix_snapshots_locked(&mut self) {
        self.scratch_terminal_sequences.clear();
        for slot in &self.slot_scheduler.slots {
            if slot.seq_id >= 0 {
                match slot.phase {
                    SlotPhase::Completed => {
                        self.scratch_terminal_sequences.push((slot.seq_id, true))
                    }
                    SlotPhase::Failed => self.scratch_terminal_sequences.push((slot.seq_id, false)),
                    _ => {}
                }
            }
        }

        for i in 0..self.scratch_terminal_sequences.len() {
            let (seq_id, completed) = self.scratch_terminal_sequences[i];
            if completed {
                self.prefix_state_cache
                    .drain_best_pending_snapshot_for_seq(self.shared_context, seq_id);
            } else {
                self.prefix_state_cache
                    .drop_pending_snapshots_for_seq(seq_id);
            }
        }
    }
}
