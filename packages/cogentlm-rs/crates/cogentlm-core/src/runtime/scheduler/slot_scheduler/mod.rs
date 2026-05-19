//! Slot scheduler: admits queued requests into free slots and picks which slots run prefill vs. decode each tick.

use crate::runtime::config::{SchedulerPolicyConfig, SchedulerTickBudget};

use super::{SlotPhase, SlotState};

mod budget;
mod flow;
mod metrics;

#[cfg(test)]
mod tests {
    mod budget_tests;
    mod flow_tests;
    mod metrics_tests;
}

#[derive(Debug, Default)]
pub struct SlotScheduler {
    slots: Vec<SlotState>,
}

impl SlotScheduler {
    pub fn slots(&self) -> &[SlotState] {
        &self.slots
    }

    pub fn mutable_slots(&mut self) -> &mut [SlotState] {
        &mut self.slots
    }

    pub fn build_tick_budget(
        policy: SchedulerPolicyConfig,
        decode_ready_count: i32,
        prefill_ready_count: i32,
        max_batch_tokens: i32,
        _prefill_chunk_size: i32,
    ) -> SchedulerTickBudget {
        budget::build_tick_budget(
            policy,
            decode_ready_count,
            prefill_ready_count,
            max_batch_tokens,
        )
    }

    pub fn is_idle(&self) -> bool {
        self.slots.iter().all(|slot| slot.phase == SlotPhase::Idle)
    }
}
