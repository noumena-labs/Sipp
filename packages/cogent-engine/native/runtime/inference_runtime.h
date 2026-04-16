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
#include <unordered_set>
#include <utility>

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
struct mtmd_context;

namespace noumena::cogentengine {

enum class RequestStepResult : std::int32_t {
  Invalid = -1,
  FatalNoProgress = -2,
  Waiting = 0,
  Progressed = 1,
  Terminal = 2,
};

struct SchedulerBurstResult {
  RequestStepResult status = RequestStepResult::Waiting;
  int32_t ticks_executed = 0;
  int32_t progressed_ticks = 0;
  int32_t completed_response_count = 0;
  int32_t emitted_token_count = 0;
};

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
  void ResetRuntimeObservability();


  GenerateRequestId EnqueueRequest(std::string context_key, std::string prompt,
                                   int n_tokens_predict,
                                   TokenCallback on_token_received = {});
  GenerateRequestId EnqueueMultimodalRequest(
      std::string context_key, std::string prompt, int n_tokens_predict,
      std::vector<std::pair<const std::uint8_t *, std::size_t>> image_views,
      TokenCallback on_token_received = {});
  bool CancelRequest(GenerateRequestId request_id);
  RequestStepResult RunSchedulerTick();
  SchedulerBurstResult RunSchedulerBurst(int32_t max_ticks,
                                         int32_t max_completed_responses,
                                         int32_t max_emitted_tokens,
                                         int32_t max_duration_us = 0);
  RequestStepResult RunRequestStep(GenerateRequestId request_id);
  std::vector<GenerateRequestId> DrainCompletedResponseIds(int32_t max_count);
  std::vector<RuntimeEvent> DrainRuntimeEvents(int32_t max_count,
                                               int32_t max_text_bytes);
  bool TryPeekCompletedResponse(GenerateRequestId request_id,
                                GenerateResponse &out_response) const;
  bool ConsumeCompletedResponse(GenerateRequestId request_id);
  const char *GetMediaMarker() const;
  const char *GetChatTemplate() const;
  std::string ApplyChatTemplate(
      const std::vector<llama_chat_message> &messages,
      bool add_assistant) const;

private:
  bool EnsureContextSpace(SequenceState &state, int new_tokens_needed,
                          int n_ctx);
  int32_t ResolveInitialDecodeContextReservationLocked(
      int32_t max_output_tokens) const;
  bool EnsureDecodeStepContextSpaceLocked(SlotState &slot);
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
  bool RunMultimodalPrefillLocked(SlotState &slot,
                                  const llama_vocab *vocab);

  bool RunPolicyBatchTickLocked();
  RequestStepResult RunSchedulerTickLocked();
  int32_t ResolvePrefillChunkSizeLocked(
      const SchedulerTickBudget &tick_budget, int32_t decode_ready_count,
      int32_t prefill_ready_count) const;
  void UpdateSharedBatchMetricsLocked(const SharedBatchPlan &plan);
  void UpdateSchedulerObservabilityLocked(const SharedBatchPlan &plan,
                                          const SchedulerTickBudget &budget,
                                          int32_t effective_prefill_chunk_size);
  void CommitNewCompletedResponsesObservabilityLocked();
  void CommitCompletedObservabilityLocked(GenerateRequestId request_id,
                                          const GenerateResponse &response);
  llama_context *CreateContext() const;

  InferenceRuntimeConfig config_;
  llama_model *primary_model_ = nullptr;
  llama_context *shared_context_ = nullptr;
  llama_sampler *sampler_ = nullptr;
  mtmd_context *mtmd_ctx_ = nullptr;
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
  std::unordered_set<GenerateRequestId> committed_observability_request_ids_;
  mutable std::mutex operation_mutex_;
};

} // namespace noumena::cogentengine
