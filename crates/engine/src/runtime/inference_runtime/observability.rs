use crate::collection::sorted_copied_values;
use crate::runtime::metrics::{CacheSource, RuntimeObservabilityMetrics};

use super::{clamp_usize_to_i32, InferenceRuntime};

impl InferenceRuntime {
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
        if self.request_queue.completed_responses.len()
            == self.committed_observability_request_ids.len()
        {
            return;
        }
        let completed_request_ids =
            sorted_copied_values(self.request_queue.completed_responses.keys().copied());
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

            self.committed_observability_request_ids.insert(request_id);

            let committed_metrics = {
                let Some(response) = self.request_queue.completed_responses.get_mut(&request_id)
                else {
                    continue;
                };
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
        self.slot_scheduler.slots.iter().find_map(|slot| {
            slot.request()
                .map(RuntimeObservabilityMetrics::from_request)
        })
    }

    fn total_observability(&self) -> RuntimeObservabilityMetrics {
        RuntimeObservabilityMetrics {
            input_tokens: clamp_usize_to_i32(self.total_input_tokens),
            output_tokens: clamp_usize_to_i32(self.total_output_tokens),
            cache_mode: self.config.cache.mode.into(),
            cache_source: CacheSource::None,
            cache_hits: clamp_usize_to_i32(self.total_cache_hits),
            prefill_tokens: clamp_usize_to_i32(self.total_prefill_tokens),
            prefill_ms: self.total_prefill_ms,
            decode_ms: self.total_decode_ms,
            ..RuntimeObservabilityMetrics::default()
        }
    }
}
