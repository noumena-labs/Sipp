/////////////////////////////////////////////////////////////////////////////////////////////////
//
// ffi_types.h
//
// - Minimal FFI surface for inference-only callbacks.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <stdint.h>

typedef struct CE_PromptPerfMetrics {
  double total_ms;
  double prompt_eval_ms;
  double decode_eval_ms;
  double sample_ms;
  int32_t prompt_eval_tokens;
  int32_t decode_eval_count;
  int32_t sample_count;
  int32_t output_token_count;
} CE_PromptPerfMetrics;
