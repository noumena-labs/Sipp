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
#include "runtime/llama/llama_batch_builder.h"
#include "runtime/metrics/observability_metrics.h"
#include "runtime/request/request_queue.h"
#include "runtime/scheduler/batch_planner.h"
#include "runtime/scheduler/slot_scheduler.h"
#include "runtime/session/prefix_cache_policy.h"
#include "runtime/session/prefix_state_cache.h"
#include "runtime/session/session_store.h"

struct llama_model;
struct llama_sampler;

namespace noumena::cogentengine {

class InferenceRuntime {
public:
  using TokenCallback = std::function<bool(const char *, int32_t)>;

  explicit InferenceRuntime(std::string model_path,
                            InferenceRuntimeConfig config = {});
  ~InferenceRuntime();

  bool IsReady() const;
  bool TryGetRuntimeObservability(RuntimeObservabilityMetrics &out) const;
  bool RuntimeObservabilityEnabled() const;
  bool BackendProfilingEnabled() const;

  bool Prompt(std::string context_key, std::string prompt, int n_tokens_predict,
              TokenCallback on_token_received = {});

  GenerateRequestId EnqueueRequest(std::string context_key, std::string prompt,
                                   int n_tokens_predict,
                                   TokenCallback on_token_received = {});
  bool CancelRequest(GenerateRequestId request_id);
  bool RunUntilRequestCompletes(GenerateRequestId request_id,
                                GenerateResponse &out_response);

private:
  bool EnsureContextSpace(SequenceState &state, int new_tokens_needed,
                          int n_ctx);
  bool PrepareSequenceForPromptLocked(const std::string &context_key,
                                      const std::vector<llama_token> &prompt_tokens,
                                      int n_tokens_predict,
                                      SequenceState &state,
                                      GenerateRequest *request,
                                      std::size_t &out_prefill_cursor);
  void MaybeStorePrefixCacheEntryLocked(const std::string &context_key,
                                        const SequenceState &state,
                                        std::size_t token_count,
                                        std::size_t terminal_token_count,
                                        GenerateRequest *request);
  bool ExecutePromptTokensLocked(const std::string &context_key,
                                 const std::vector<llama_token> &prompt_tokens,
                                 int n_tokens_predict,
                                 TokenCallback on_token_received);
  bool ExecuteSingleSlotRequestLocked(SlotState &slot);
  bool RunSharedBatchTickLocked();
  bool RunPolicyBatchTickLocked();
  void UpdateSharedBatchMetricsLocked(const SharedBatchPlan &plan);
  void UpdateSchedulerObservabilityLocked(const SharedBatchPlan &plan,
                                          const SchedulerTickBudget &budget);
  llama_context *CreateContext() const;

  InferenceRuntimeConfig config_;
  llama_model *primary_model_ = nullptr;
  llama_context *shared_context_ = nullptr;
  llama_sampler *sampler_ = nullptr;
  RuntimeObservabilityMetrics last_runtime_observability_;
  bool has_last_runtime_observability_ = false;
  SessionStore session_store_;
  RequestQueue request_queue_;
  SlotScheduler slot_scheduler_;
  BatchPlanner batch_planner_;
  LlamaBatchBuilder shared_batch_builder_;
  SharedBatchObservabilityMetrics shared_batch_observability_;
  SchedulerObservabilityMetrics scheduler_observability_;
  PrefixStateCache prefix_state_cache_;
  PrefixCachePolicy prefix_cache_policy_;
  GenerateRequestId next_request_id_ = 1;
  std::uint64_t model_fingerprint_ = 0;
  mutable std::mutex operation_mutex_;
};

} // namespace noumena::cogentengine
