/////////////////////////////////////////////////////////////////////////////////////////////////
//
// observability_metrics.h
//
// - Runtime observability DTOs.
// - These metrics are optional and should stay outside the core inference contract.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

namespace noumena::cogentengine {

struct RuntimeObservabilityMetrics {
  double total_ms = 0.0;
  double prompt_eval_ms = 0.0;
  double decode_eval_ms = 0.0;
  double sample_ms = 0.0;
  double queue_delay_ms = 0.0;
  double ttft_ms = 0.0;
  double mean_itl_ms = 0.0;
  double tail_itl_ms = 0.0;
  double e2e_ms = 0.0;
  double native_scheduler_tick_ms = 0.0;
  double native_scheduler_admit_ms = 0.0;
  double native_scheduler_finalize_ms = 0.0;
  double native_scheduler_commit_ms = 0.0;
  double native_policy_prepare_ms = 0.0;
  double native_policy_plan_ms = 0.0;
  double native_batch_build_ms = 0.0;
  double native_llama_decode_wall_ms = 0.0;
  double native_synchronize_ms = 0.0;
  double native_kv_update_ms = 0.0;
  double native_sampler_wall_ms = 0.0;
  double native_token_emit_ms = 0.0;
  double native_prefix_cache_ms = 0.0;
  double native_observability_ms = 0.0;
  int32_t input_token_count = 0;
  int32_t prompt_eval_tokens = 0;
  int32_t decode_eval_count = 0;
  int32_t sample_count = 0;
  int32_t output_token_count = 0;
  int32_t first_sampled_token_id = -1;
  int32_t batch_participation_count = 0;
  int32_t decode_first_tick_count = 0;
  int32_t chunked_prefill_tick_count = 0;
  int32_t mixed_workload_tick_count = 0;
  int32_t lcp_reuse_tokens = 0;
  int32_t prefix_cache_restore_tokens = 0;
  int32_t prefix_cache_hit_count = 0;
  int32_t prefix_cache_store_count = 0;
  int32_t native_policy_tick_count = 0;
  int32_t native_scheduler_tick_count = 0;
};

struct SharedBatchObservabilityMetrics {
  std::uint64_t tick_count = 0;
  std::uint64_t total_occupied_slots = 0;
  std::uint64_t total_prefill_tokens = 0;
  std::uint64_t total_decode_tokens = 0;
};

struct SchedulerObservabilityMetrics {
  std::uint64_t tick_count = 0;
  std::uint64_t decode_first_tick_count = 0;
  std::uint64_t chunked_prefill_tick_count = 0;
  std::uint64_t mixed_workload_tick_count = 0;
  double accumulated_queue_delay_ms = 0.0;
  double accumulated_ttft_ms = 0.0;
  double max_tail_itl_ms = 0.0;
};

} // namespace noumena::cogentengine
