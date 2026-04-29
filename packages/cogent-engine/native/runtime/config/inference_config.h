/////////////////////////////////////////////////////////////////////////////////////////////////
//
// inference_config.h
//
// - Shared runtime configuration surface.
// - Keep this aligned with src/types.ts and native/api/ffi_types.h.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <string>

#include "runtime/config/scheduler_policy.h"

namespace noumena::cogentengine {

struct InferenceRuntimeConfig {
  int32_t n_ctx = 0;
  int32_t n_batch = 0;
  int32_t n_ubatch = 0;
  int32_t n_seq_max = 1;
  int32_t n_threads = 0;
  int32_t n_threads_batch = 0;
  int32_t gpu_layers = -1;
  int32_t flash_attention = -1;
  int32_t kv_unified = -1;
  int32_t max_cached_sessions = 8;
  int32_t retained_prefix_tokens = 100;
  int32_t prefill_chunk_size = 0;
  int32_t prefix_cache_interval_tokens = 128;
  int32_t max_prefix_cache_entries = 32;
  std::string mmproj_path;
  int32_t multimodal_use_gpu = -1;
  int32_t debug_compare_multimodal_embeddings = 0;
  int32_t image_min_tokens = 0;
  int32_t image_max_tokens = 0;
  int32_t sampling_repeat_last_n = 64;
  float sampling_repeat_penalty = 1.05f;
  float sampling_frequency_penalty = 0.0f;
  float sampling_presence_penalty = 0.0f;
  int32_t sampling_top_k = 40;
  float sampling_top_p = 0.8f;
  float sampling_min_p = 0.0f;
  float sampling_temperature = 0.7f;
  int32_t sampling_seed = -1;
  SchedulerPolicyConfig scheduler_policy{};
  int32_t enable_runtime_observability = 0;
  int32_t enable_backend_profiling = 0;
};

} // namespace noumena::cogentengine
