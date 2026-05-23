use std::time::{Duration, Instant};

use crate::runtime::config::SchedulerTickBudget;
use crate::runtime::numeric::positive_fair_share_i32;

use super::{
    saturating_i32_delta, saturating_usize_delta_to_i32, InferenceRuntime, RequestStepResult,
    SchedulerBurstResult, PREFIX_SNAPSHOT_COMMIT_BUDGET,
};

impl InferenceRuntime {
    pub fn run_scheduler_tick(&mut self) -> RequestStepResult {
        self.run_scheduler_tick_locked()
    }

    pub fn run_scheduler_burst(
        &mut self,
        max_ticks: i32,
        max_completed_responses: i32,
        max_emitted_tokens: i32,
        max_duration: Duration,
    ) -> SchedulerBurstResult {
        let mut burst_result = SchedulerBurstResult::default();
        if max_ticks <= 0 || !self.is_ready() {
            burst_result.status = RequestStepResult::Invalid;
            return burst_result;
        }

        let max_completed = max_completed_responses.max(0);
        let max_emitted = max_emitted_tokens.max(0);
        let deadline = (!max_duration.is_zero()).then(|| Instant::now() + max_duration);

        for _ in 0..max_ticks {
            let completed_before = self.request_queue.completed_response_count();
            let emitted_before = self.request_queue.total_emitted_token_count();
            let step_result = self.run_scheduler_tick_locked();
            let completed_after = self.request_queue.completed_response_count();
            let emitted_after = self.request_queue.total_emitted_token_count();

            record_tick_progress(
                &mut burst_result,
                completed_before,
                completed_after,
                emitted_before,
                emitted_after,
                step_result,
            );

            if matches!(
                step_result,
                RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
            ) {
                burst_result.status = step_result;
                return burst_result;
            }

            if step_result == RequestStepResult::Waiting {
                self.commit_pending_prefix_snapshots();
                burst_result.status = completed_or_waiting(&burst_result);
                return burst_result;
            }

            let completed_limit_reached =
                max_completed > 0 && burst_result.completed_response_count >= max_completed;
            let emitted_limit_reached =
                max_emitted > 0 && burst_result.emitted_token_count >= max_emitted;
            let duration_limit_reached =
                deadline.is_some_and(|deadline| Instant::now() >= deadline);

            if completed_limit_reached || emitted_limit_reached || duration_limit_reached {
                if burst_result.completed_response_count > 0 {
                    self.commit_pending_prefix_snapshots();
                }
                burst_result.status = completed_or_waiting(&burst_result);
                return burst_result;
            }
        }

        if burst_result.completed_response_count > 0 {
            self.commit_pending_prefix_snapshots();
        }
        burst_result.status = completed_or_waiting(&burst_result);
        burst_result
    }

    pub fn run_scheduler_loop(
        &mut self,
        max_ticks: i32,
        max_completed_responses: i32,
        max_emitted_tokens: i32,
        max_duration: Duration,
    ) -> SchedulerBurstResult {
        let mut loop_result = SchedulerBurstResult::default();
        if !self.is_ready() {
            loop_result.status = RequestStepResult::Invalid;
            return loop_result;
        }

        let loop_start = Instant::now();
        loop {
            if self.request_queue.live_request_count() == 0 {
                loop_result.status = RequestStepResult::Waiting;
                break;
            }

            let completed_before = self.request_queue.completed_response_count();
            let emitted_before = self.request_queue.total_emitted_token_count();
            let step_result = self.run_scheduler_tick_locked();
            let completed_after = self.request_queue.completed_response_count();
            let emitted_after = self.request_queue.total_emitted_token_count();

            record_tick_progress(
                &mut loop_result,
                completed_before,
                completed_after,
                emitted_before,
                emitted_after,
                step_result,
            );

            if matches!(
                step_result,
                RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
            ) {
                loop_result.status = step_result;
                break;
            }

            if max_ticks > 0 && loop_result.ticks_executed >= max_ticks {
                loop_result.status = RequestStepResult::Progressed;
                break;
            }
            if max_completed_responses > 0
                && loop_result.completed_response_count >= max_completed_responses
            {
                loop_result.status = RequestStepResult::Progressed;
                break;
            }
            if max_emitted_tokens > 0 && loop_result.emitted_token_count >= max_emitted_tokens {
                loop_result.status = RequestStepResult::Progressed;
                break;
            }
            if !max_duration.is_zero() && loop_start.elapsed() >= max_duration {
                loop_result.status = completed_or_waiting(&loop_result);
                break;
            }
            if step_result == RequestStepResult::Waiting {
                loop_result.status = completed_or_waiting(&loop_result);
                break;
            }
        }

        if loop_result.completed_response_count > 0
            || loop_result.status == RequestStepResult::Waiting
        {
            self.commit_pending_prefix_snapshots();
        }
        loop_result
    }

    fn commit_pending_prefix_snapshots(&mut self) {
        if !self.slot_scheduler.is_idle() {
            return;
        }
        self.prefix_state_cache
            .drain_pending_snapshots(self.shared_context, PREFIX_SNAPSHOT_COMMIT_BUDGET);
    }

    pub(super) fn resolve_prefill_chunk_size_locked(
        &self,
        tick_budget: SchedulerTickBudget,
        decode_ready_count: i32,
        prefill_ready_count: i32,
    ) -> i32 {
        let configured_chunk_size = self.config.scheduler.prefill_chunk_size.max(0);
        if !self
            .config
            .scheduler
            .policy
            .enable_adaptive_prefill_chunking
            || prefill_ready_count <= 0
        {
            return configured_chunk_size;
        }

        if decode_ready_count <= 0 && configured_chunk_size <= 0 {
            return 0;
        }

        let prefill_budget = tick_budget.effective_prefill_budget();
        if prefill_budget <= 0 {
            return configured_chunk_size;
        }

        let fair_share = positive_fair_share_i32(prefill_budget, prefill_ready_count);
        if configured_chunk_size > 0 {
            configured_chunk_size.min(fair_share)
        } else {
            fair_share
        }
    }
}

fn record_tick_progress(
    result: &mut SchedulerBurstResult,
    completed_before: usize,
    completed_after: usize,
    emitted_before: i32,
    emitted_after: i32,
    step_result: RequestStepResult,
) {
    result.ticks_executed = result.ticks_executed.saturating_add(1);
    if completed_after > completed_before {
        result.completed_response_count =
            result
                .completed_response_count
                .saturating_add(saturating_usize_delta_to_i32(
                    completed_after,
                    completed_before,
                ));
    }
    if emitted_after > emitted_before {
        result.emitted_token_count = result
            .emitted_token_count
            .saturating_add(saturating_i32_delta(emitted_after, emitted_before));
    }
    if matches!(
        step_result,
        RequestStepResult::Progressed | RequestStepResult::Terminal
    ) {
        result.progressed_ticks = result.progressed_ticks.saturating_add(1);
    }
}

pub(super) fn completed_or_waiting(result: &SchedulerBurstResult) -> RequestStepResult {
    if result.progressed_ticks > 0 || result.completed_response_count > 0 {
        RequestStepResult::Progressed
    } else {
        RequestStepResult::Waiting
    }
}
