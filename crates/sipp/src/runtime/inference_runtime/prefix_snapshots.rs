use crate::runtime::config::KvReuseMode;
use crate::runtime::scheduler::BatchContributionKind;

use super::{unique_slot_first_use, InferenceRuntime};

impl InferenceRuntime {
    pub(super) fn capture_prefix_snapshots(
        &mut self,
        plan: &crate::runtime::scheduler::SharedBatchPlan,
    ) {
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
            let Some(terminal_token_count) =
                decode_seed_snapshot_token_count(request.prompt_tokens.len())
            else {
                continue;
            };
            if slot.mirror.current_kv_tokens.len() > terminal_token_count {
                continue;
            }
            // Pure snapshot mode evicts the sequence at completion, so it must
            // materialize before finalization. Live+snapshot keeps the
            // sequence idle and can defer the expensive state readback.
            if request.cache_mode == KvReuseMode::StateSnapshot {
                self.kv_cache.capture_prefix_snapshot(
                    &self.native_runtime,
                    self.model_fingerprint,
                    &request.context_key,
                    slot.seq_id,
                    &slot.mirror.current_kv_tokens,
                    terminal_token_count,
                );
            } else {
                self.kv_cache.queue_prefix_snapshot(
                    self.model_fingerprint,
                    &request.context_key,
                    slot.seq_id,
                    slot.lease_generation,
                    &slot.mirror.current_kv_tokens,
                    terminal_token_count,
                );
            }
        }
    }
}

pub(super) fn decode_seed_snapshot_token_count(prompt_len: usize) -> Option<usize> {
    prompt_len
        .checked_sub(1)
        .filter(|&token_count| token_count > 0)
}

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/prefix_snapshots_tests.rs"]
mod prefix_snapshots_tests;
