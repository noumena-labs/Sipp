/////////////////////////////////////////////////////////////////////////////////////////////////
//
// inference_runtime.h
//
// - Inference-only runtime over llama.cpp.
// - Owns model lifetime, context reuse, and text generation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

#include "runtime/config/inference_config.h"
#include "runtime/request/request_queue.h"
#include "runtime/scheduler/slot_scheduler.h"
#include "runtime/session/session_store.h"

struct llama_model;
struct llama_sampler;

namespace noumena::cogentengine {

struct PromptPerfStats {
  double total_ms = 0.0;
  double prompt_eval_ms = 0.0;
  double decode_eval_ms = 0.0;
  double sample_ms = 0.0;
  int32_t input_token_count = 0;
  int32_t prompt_eval_tokens = 0;
  int32_t decode_eval_count = 0;
  int32_t sample_count = 0;
  int32_t output_token_count = 0;
};

class InferenceRuntime {
public:
  using TokenCallback = std::function<void(const char *, int32_t)>;

  explicit InferenceRuntime(std::string model_path,
                            InferenceRuntimeConfig config = {});
  ~InferenceRuntime();

  bool IsReady() const;
  bool TryGetLastPromptPerf(PromptPerfStats &out) const;

  bool Prompt(std::string context_key, std::string prompt, int n_tokens_predict,
              TokenCallback on_token_received = {});

  GenerateRequestId EnqueueRequest(std::string context_key, std::string prompt,
                                   int n_tokens_predict,
                                   TokenCallback on_token_received = {});
  bool RunUntilRequestCompletes(GenerateRequestId request_id,
                                GenerateResponse &out_response);

private:
  bool EnsureContextSpace(ContextState &state, int new_tokens_needed,
                          int n_ctx);
  bool ExecutePromptTokensLocked(const std::string &context_key,
                                 const std::vector<llama_token> &prompt_tokens,
                                 int n_tokens_predict,
                                 TokenCallback on_token_received);
  llama_context *CreateContext() const;

  InferenceRuntimeConfig config_;
  llama_model *primary_model_ = nullptr;
  llama_sampler *sampler_ = nullptr;
  PromptPerfStats last_prompt_perf_;
  bool has_last_prompt_perf_ = false;
  SessionStore session_store_;
  RequestQueue request_queue_;
  SlotScheduler slot_scheduler_;
  GenerateRequestId next_request_id_ = 1;
  mutable std::mutex operation_mutex_;
};

} // namespace noumena::cogentengine
