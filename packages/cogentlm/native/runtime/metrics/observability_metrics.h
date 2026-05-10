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
  int32_t input_token_count = 0;
  int32_t prompt_eval_tokens = 0;
  int32_t decode_eval_count = 0;
  int32_t sample_count = 0;
  int32_t output_token_count = 0;
  int32_t first_sampled_token_id = -1;
  int32_t batch_participation_count = 0;
};

} // namespace noumena::cogentengine
