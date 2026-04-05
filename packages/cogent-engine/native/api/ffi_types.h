/////////////////////////////////////////////////////////////////////////////////////////////////
//
// ffi_types.h
//
// - Minimal FFI surface for inference-only callbacks.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <stdint.h>

typedef uint64_t CE_RequestId;
typedef int32_t (*CE_TokenCallback)(const char *token_piece,
                                    int32_t token_length);

typedef struct CE_InitConfig {
  int32_t n_ctx;
  int32_t n_batch;
  int32_t n_ubatch;
  int32_t n_seq_max;
  int32_t n_threads;
  int32_t n_threads_batch;
  int32_t gpu_layers;
  int32_t flash_attention;
  int32_t kv_unified;
  int32_t max_cached_sessions;
  int32_t retained_prefix_tokens;
  int32_t prefill_chunk_size;
  int32_t prefix_cache_interval_tokens;
  int32_t max_prefix_cache_entries;
  int32_t scheduler_policy;
  int32_t decode_token_reserve;
  int32_t adaptive_prefill_chunking;
} CE_InitConfig;

typedef struct CE_PromptPerfMetrics {
  double total_ms;
  double prompt_eval_ms;
  double decode_eval_ms;
  double sample_ms;
  double queue_delay_ms;
  double ttft_ms;
  double mean_itl_ms;
  double tail_itl_ms;
  double e2e_ms;
  int32_t input_token_count;
  int32_t prompt_eval_tokens;
  int32_t decode_eval_count;
  int32_t sample_count;
  int32_t output_token_count;
  int32_t scheduler_tick_count;
  int32_t batch_participation_count;
  int32_t decode_first_tick_count;
  int32_t chunked_prefill_tick_count;
  int32_t mixed_workload_tick_count;
  int32_t lcp_reuse_tokens;
  int32_t prefix_cache_restore_tokens;
  int32_t prefix_cache_hit_count;
  int32_t prefix_cache_store_count;
} CE_PromptPerfMetrics;
