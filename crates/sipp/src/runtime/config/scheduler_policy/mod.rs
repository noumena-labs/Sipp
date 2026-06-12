//! Scheduler policy knobs (balanced vs. decode-first, adaptive prefill chunking).

use serde::{Deserialize, Serialize};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../../tests/runtime/config/scheduler_policy_tests.rs"]
mod scheduler_policy_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const DEFAULT_DECODE_TOKEN_RESERVE: i32 = 1;
const DEFAULT_ADAPTIVE_PREFILL_CHUNKING: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerPolicyMode {
    LatencyFirst = 0,
    #[default]
    Balanced = 1,
    ThroughputFirst = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerPolicyConfig {
    pub mode: SchedulerPolicyMode,
    pub decode_token_reserve: i32,
    pub enable_adaptive_prefill_chunking: bool,
}

impl Default for SchedulerPolicyConfig {
    fn default() -> Self {
        Self {
            mode: SchedulerPolicyMode::default(),
            decode_token_reserve: DEFAULT_DECODE_TOKEN_RESERVE,
            enable_adaptive_prefill_chunking: DEFAULT_ADAPTIVE_PREFILL_CHUNKING,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SchedulerTickBudget {
    pub total_token_budget: i32,
    pub reserved_decode_tokens: i32,
    pub reserved_prefill_tokens: i32,
    pub decode_first: bool,
}

impl SchedulerTickBudget {
    pub fn effective_decode_budget(&self) -> i32 {
        self.reserved_decode_tokens
            .clamp(0, self.total_token_budget)
    }

    pub fn effective_prefill_budget(&self) -> i32 {
        self.reserved_prefill_tokens
            .clamp(0, self.total_token_budget - self.effective_decode_budget())
    }
}
