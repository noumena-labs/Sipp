use std::time::Instant;

use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::GenerateRequest;
use crate::runtime::scheduler::BatchContributionKind;

use super::{
    clamp_usize_to_i32, duration_ms, unique_slot_first_use, DebugMetricsTick, InferenceRuntime,
};

impl InferenceRuntime {
    pub fn runtime_observability_enabled(&self) -> bool {
        self.config.observability.effective_runtime_metrics()
    }

    pub fn backend_profiling_enabled(&self) -> bool {
        self.config.observability.backend_profiling
    }

    pub fn try_get_runtime_observability(&self) -> Option<RuntimeObservabilityMetrics> {
        if !self.config.observability.effective_runtime_metrics() {
            return None;
        }

        self.active_request_observability()
            .or_else(|| {
                self.has_last_runtime_observability
                    .then_some(self.last_runtime_observability)
            })
            .or_else(|| Some(self.total_observability()))
    }

    pub(super) fn commit_new_completed_responses_observability_locked(&mut self) {
        // Counts move in lockstep, so avoid allocating ids when nothing new finished.
        if self.request_queue.completed_response_count()
            == self.committed_observability_request_ids.len()
        {
            return;
        }
        let completed_request_ids = self.request_queue.completed_response_ids();
        if completed_request_ids.is_empty() {
            return;
        }

        for request_id in completed_request_ids {
            if request_id == 0
                || self
                    .committed_observability_request_ids
                    .contains(&request_id)
            {
                continue;
            }

            let debug_metrics_commit_start = Instant::now();
            self.committed_observability_request_ids.insert(request_id);

            let committed_metrics = {
                let Some(response) = self.request_queue.find_mut_completed_response(request_id)
                else {
                    continue;
                };
                response
                    .runtime_observability
                    .debug_metrics_commit_observability_ms +=
                    duration_ms(debug_metrics_commit_start, Instant::now());
                self.config
                    .observability
                    .effective_runtime_metrics()
                    .then_some(response.runtime_observability)
            };

            if let Some(metrics) = committed_metrics {
                self.last_runtime_observability = metrics;
                self.has_last_runtime_observability = true;
            }
        }
    }

    fn active_request_observability(&self) -> Option<RuntimeObservabilityMetrics> {
        self.slot_scheduler
            .slots()
            .iter()
            .find_map(|slot| slot.request().map(request_observability))
    }

    fn total_observability(&self) -> RuntimeObservabilityMetrics {
        RuntimeObservabilityMetrics {
            input_tokens: clamp_usize_to_i32(self.total_input_tokens),
            output_tokens: clamp_usize_to_i32(self.total_output_tokens),
            cache_hits: clamp_usize_to_i32(self.total_cache_hits),
            prefill_tokens: clamp_usize_to_i32(self.total_prefill_tokens),
            prefill_ms: self.total_prefill_ms,
            decode_ms: self.total_decode_ms,
            ..RuntimeObservabilityMetrics::default()
        }
    }

    pub(super) fn apply_debug_metrics_to_plan(
        &mut self,
        plan: &crate::runtime::scheduler::SharedBatchPlan,
        debug_metrics: DebugMetricsTick,
    ) {
        let mut timed_slots: u64 = 0;
        let mut decode_slots: u64 = 0;
        let mut prefill_slots: u64 = 0;

        for contribution in &plan.contributions {
            let Some(slot) = self
                .slot_scheduler
                .mutable_slots()
                .get_mut(contribution.slot_index)
            else {
                continue;
            };
            let Some(request) = slot.request_mut() else {
                continue;
            };

            if unique_slot_first_use(&mut timed_slots, contribution.slot_index) {
                apply_debug_metrics_tick(request, debug_metrics);
            }

            match contribution.kind {
                BatchContributionKind::Prefill => {
                    if unique_slot_first_use(&mut prefill_slots, contribution.slot_index) {
                        increment_debug_counter(&mut request.debug_metrics_prefill_ticks);
                    }
                }
                BatchContributionKind::Decode => {
                    if unique_slot_first_use(&mut decode_slots, contribution.slot_index) {
                        increment_debug_counter(&mut request.debug_metrics_decode_ticks);
                    }
                }
            }
        }
    }
}

fn apply_debug_metrics_tick(request: &mut GenerateRequest, debug_metrics: DebugMetricsTick) {
    increment_debug_counter(&mut request.debug_metrics_scheduler_ticks);
    request.debug_metrics_normalize_ms += debug_metrics.normalize_ms;
    request.debug_metrics_select_slots_ms += debug_metrics.select_slots_ms;
    request.debug_metrics_plan_ms += debug_metrics.plan_ms;
    request.debug_metrics_batch_build_ms += debug_metrics.batch_build_ms;
    request.debug_metrics_llama_decode_ms += debug_metrics.llama_decode_ms;
    request.debug_metrics_llama_sync_ms += debug_metrics.llama_sync_ms;
    request.debug_metrics_apply_bookkeeping_ms += debug_metrics.apply_bookkeeping_ms;
    request.debug_metrics_apply_decode_results_ms += debug_metrics.apply_decode_results_ms;
    request.debug_metrics_sample_ms += debug_metrics.sample_ms;
    request.debug_metrics_token_piece_ms += debug_metrics.token_piece_ms;
    request.debug_metrics_emit_ms += debug_metrics.emit_ms;
    request.debug_metrics_prefix_queue_ms += debug_metrics.prefix_queue_ms;
    request.debug_metrics_post_decode_ms += debug_metrics.post_decode_ms;
}

fn request_observability(request: &GenerateRequest) -> RuntimeObservabilityMetrics {
    RuntimeObservabilityMetrics::from_request(request)
}

fn increment_debug_counter(counter: &mut i32) {
    *counter = counter.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::increment_debug_counter;

    #[test]
    fn increment_debug_counter_saturates() {
        let mut counter = i32::MAX;

        increment_debug_counter(&mut counter);

        assert_eq!(counter, i32::MAX);
    }
}
