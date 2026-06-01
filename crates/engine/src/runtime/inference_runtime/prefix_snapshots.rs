use crate::runtime::scheduler::BatchContributionKind;

use super::{unique_slot_first_use, InferenceRuntime};

impl InferenceRuntime {
    pub(super) fn capture_prefix_snapshots(
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
            let Some(terminal_token_count) =
                decode_seed_snapshot_token_count(request.prompt_tokens.len())
            else {
                continue;
            };
            if slot.mirror.current_kv_tokens.len() > terminal_token_count {
                continue;
            }
            self.kv_cache.capture_prefix_snapshot(
                &self.native_runtime,
                self.model_fingerprint,
                &request.context_key,
                slot.seq_id,
                &slot.mirror.current_kv_tokens,
                terminal_token_count,
            );
        }
    }
}

pub(super) fn decode_seed_snapshot_token_count(prompt_len: usize) -> Option<usize> {
    prompt_len
        .checked_sub(1)
        .filter(|&token_count| token_count > 0)
}

#[cfg(test)]
mod tests {
    use super::decode_seed_snapshot_token_count;

    #[test]
    fn decode_seed_snapshot_requires_at_least_two_prompt_tokens() {
        assert_eq!(decode_seed_snapshot_token_count(0), None);
        assert_eq!(decode_seed_snapshot_token_count(1), None);
        assert_eq!(decode_seed_snapshot_token_count(2), Some(1));
        assert_eq!(decode_seed_snapshot_token_count(19), Some(18));
    }
}
