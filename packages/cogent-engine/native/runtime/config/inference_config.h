/////////////////////////////////////////////////////////////////////////////////////////////////
//
// inference_config.h
//
// - Shared Phase 1 runtime configuration surface.
// - Keep this aligned with src/types.ts and native/api/ffi_types.h.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

namespace noumena::cogentengine {

struct InferenceRuntimeConfig {
  int32_t n_ctx = 0;
  int32_t n_batch = 0;
  int32_t n_ubatch = 0;
  int32_t n_seq_max = 1;
  int32_t n_threads = 0;
  int32_t n_threads_batch = 0;
  int32_t gpu_layers = 99;
  int32_t flash_attention = -1;
  int32_t kv_unified = -1;
  int32_t max_cached_sessions = 8;
  int32_t retained_prefix_tokens = 100;
};

} // namespace noumena::cogentengine
