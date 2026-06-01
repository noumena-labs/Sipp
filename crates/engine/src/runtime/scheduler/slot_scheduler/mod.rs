//! Slot scheduler: admits queued requests into free slots and picks which slots run prefill vs. decode each tick.

use crate::runtime::config::{SchedulerPolicyConfig, SchedulerTickBudget};

use super::SlotState;

mod budget;
mod flow;
mod metrics;

#[cfg(test)]
#[path = "../../../tests/runtime/scheduler/slot_scheduler/budget_tests.rs"]
mod budget_tests;

#[cfg(test)]
#[path = "../../../tests/runtime/scheduler/slot_scheduler/flow_tests.rs"]
mod flow_tests;

#[cfg(test)]
#[path = "../../../tests/runtime/scheduler/slot_scheduler/metrics_tests.rs"]
mod metrics_tests;

#[derive(Debug, Default)]
pub struct SlotScheduler {
    pub(crate) slots: Vec<SlotState>,
}

impl SlotScheduler {
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
}
