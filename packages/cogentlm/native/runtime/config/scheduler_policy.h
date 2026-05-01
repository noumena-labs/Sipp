/////////////////////////////////////////////////////////////////////////////////////////////////
//
// scheduler_policy.h
//
// - Phase 4 scheduler policy surface.
// - Keep policy selection separate from slot ownership and batch packing.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <cstdint>

namespace noumena::cogentengine {

enum class SchedulerPolicyMode : std::uint8_t {
  LatencyFirst = 0,
  Balanced = 1,
  ThroughputFirst = 2,
};

struct SchedulerPolicyConfig {
  SchedulerPolicyMode mode = SchedulerPolicyMode::Balanced;
  int32_t decode_token_reserve = 1;
  bool enable_adaptive_prefill_chunking = false;
};

struct SchedulerTickBudget {
  int32_t total_token_budget = 0;
  int32_t reserved_decode_tokens = 0;
  int32_t reserved_prefill_tokens = 0;
  bool decode_first = true;

  int32_t EffectiveDecodeBudget() const {
    return std::clamp(reserved_decode_tokens, 0, total_token_budget);
  }

  int32_t EffectivePrefillBudget() const {
    return std::clamp(reserved_prefill_tokens, 0,
                      total_token_budget - EffectiveDecodeBudget());
  }
};

} // namespace noumena::cogentengine
