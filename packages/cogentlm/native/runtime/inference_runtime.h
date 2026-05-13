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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "api/ffi_types.h"
#include "chat.h"
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
  explicit InferenceRuntime(std::string model_path,
                            InferenceRuntimeConfig config = {});
  ~InferenceRuntime();

  bool IsReady() const;
  bool TryGetRuntimeObservability(RuntimeObservabilityMetrics &out) const;
  bool RuntimeObservabilityEnabled() const;
  bool BackendProfilingEnabled() const;

  GenerateRequestId EnqueueRequest(std::string context_key, std::string prompt,
                                   int n_tokens_predict,
                                   std::string grammar = {},
                                   GenerateTokenEmissionMode token_emission_mode =
                                       GenerateTokenEmissionMode::None);
  GenerateRequestId EnqueueMultimodalRequest(
      std::string context_key, std::string prompt, int n_tokens_predict,
      std::vector<std::pair<const std::uint8_t *, std::size_t>> image_views,
      std::string grammar = {},
      GenerateTokenEmissionMode token_emission_mode =
          GenerateTokenEmissionMode::None);
  bool CancelRequest(GenerateRequestId request_id);
  RequestStepResult RunSchedulerTick();
  SchedulerBurstResult RunSchedulerBurst(int32_t max_ticks,
                                         int32_t max_completed_responses,
                                         int32_t max_emitted_tokens,
                                         int32_t max_duration_us = 0);
  // Streaming buffer accessors — see request_queue.h.  Stable addresses;
  // called once at init, then JS uses HEAP32 / HEAPU8 directly.
  const uint8_t *StreamingBufferPointer();
  std::size_t StreamingBufferCapacity();
  int32_t *StreamingBufferUsedAddress();
  int32_t *StreamingBufferDropCountAddress();
  SchedulerBurstResult RunSchedulerLoop(int32_t max_ticks,
                                        int32_t max_completed_responses,
                                        int32_t max_emitted_tokens,
                                        int32_t max_duration_us);
  bool TryPeekCompletedResponse(GenerateRequestId request_id,
                                GenerateResponse &out_response) const;
  bool HasRequest(GenerateRequestId request_id) const;
  bool ConsumeCompletedResponse(GenerateRequestId request_id);
  const char *GetMediaMarker() const;
  const char *GetChatTemplate() const;
  // Returns the model's BOS token rendered as text (empty string if none).
  std::string GetBosText() const;
  // Returns the model's EOS token rendered as text (empty string if none).
  std::string GetEosText() const;
  // Applies the model's embedded chat template to the full chat history.
  std::string ApplyChatTemplate(
      const std::vector<common_chat_msg> &messages,
      bool add_assistant) const;

private:
  bool EnsureContextSpace(SequenceState &state, llama_seq_id seq_id, int new_tokens_needed,
                          int n_ctx);
  bool ReconcilePhysicalState(SequenceState &state, llama_seq_id seq_id, llama_memory_t mem);
  int32_t ResolveInitialDecodeContextReservationLocked(
      int32_t max_output_tokens) const;
  bool EnsureDecodeStepContextSpaceLocked(SlotState &slot);
  bool PrepareSequenceForPromptLocked(const std::string &context_key,
                                      const std::vector<llama_token> &prompt_tokens,
                                      int n_tokens_predict,
                                      SequenceState &state,
                                      llama_seq_id seq_id,
                                      GenerateRequest *request,
                                      std::size_t &out_prefill_cursor);
  bool NormalizeRunnableSlotStateLocked(SlotState &slot);
  bool RecoverDecodeSeedStateLocked(SlotState &slot,
                                    GenerateRequest &request,
                                    SequenceState &session);
  std::string BuildNoProgressDiagnosticLocked() const;
  void MaybeStorePrefixCacheEntryLocked(const std::string &context_key,
                                        const SequenceState &state,
                                        llama_seq_id seq_id,
                                        std::size_t token_count,
                                        std::size_t terminal_token_count,
                                        GenerateRequest *request);
  bool RunMultimodalPrefillLocked(SlotState &slot,
                                  const llama_vocab *vocab);

  bool RunPolicyBatchTickLocked();
  void CompletePendingBookkeepingLocked();
  void EmitPendingTokensLocked();
  void FlushAllPendingBookkeepingLocked();
  RequestStepResult RunSchedulerTickLocked();
  int32_t ResolvePrefillChunkSizeLocked(
      const SchedulerTickBudget &tick_budget, int32_t decode_ready_count,
      int32_t prefill_ready_count) const;

  void CommitNewCompletedResponsesObservabilityLocked();
  void CommitCompletedObservabilityLocked(GenerateRequestId request_id,
                                          const GenerateResponse &response);
  int32_t ResolveBatchTokenBudgetLocked() const;
  llama_context *CreateContext() const;

  InferenceRuntimeConfig config_;
  llama_model *primary_model_ = nullptr;
  llama_context *shared_context_ = nullptr;
  llama_sampler *sampler_ = nullptr;
  mtmd_context *mtmd_ctx_ = nullptr;
  std::unique_ptr<common_chat_templates, common_chat_templates_deleter>
      chat_templates_;
  RuntimeObservabilityMetrics last_runtime_observability_;
  bool has_last_runtime_observability_ = false;
  SessionStore session_store_;
  RequestQueue request_queue_;
  SlotScheduler slot_scheduler_;
  BatchPlanner batch_planner_;
  LlamaBatchBuilder shared_batch_builder_;

  PrefixStateCache prefix_state_cache_;
  PrefixCachePolicy prefix_cache_policy_;
  GenerateRequestId next_request_id_ = 1;
  std::uint64_t model_fingerprint_ = 0;
  std::unordered_set<GenerateRequestId> committed_observability_request_ids_;
  std::vector<SlotState *> scratch_decode_ready_slots_;
  std::vector<SlotState *> scratch_prefill_ready_slots_;
  std::vector<SlotState *> scratch_runnable_slots_;
  std::vector<SlotState *> scratch_live_decode_ready_slots_;
  std::vector<SlotState *> scratch_live_prefill_ready_slots_;
  std::vector<SlotState *> scratch_live_runnable_slots_;
  std::vector<SlotState *> scratch_prefix_cache_slots_;
  // Slot-id-indexed seen-flags used for O(1) deduplication when populating
  // `scratch_prefix_cache_slots_`.  Sized once to slot count; reset to false
  // after each use so the memset cost is paid against a small, fixed buffer
  // rather than per-contribution std::find calls.
  std::vector<std::uint8_t> scratch_prefix_cache_seen_;
  std::vector<GenerateRequest *> scratch_tick_requests_;
  std::vector<GenerateRequest *> scratch_decode_requests_;
  std::vector<GenerateRequest *> scratch_prefill_requests_;
  // Persistent scratch for the per-tick "which batch slots produced logits we
  // need to sample from" list.  Lives across ticks so its capacity stabilizes
  // and the inference hot path performs zero heap allocations.
  struct PendingLogitsContribution {
    std::size_t contribution_index = 0;
    GenerateRequest *request = nullptr;
    int32_t batch_token_index = -1;
    llama_token sampled_token = -1;

  };

  struct PendingTickBookkeeping {
    SharedBatchPlan plan;
    std::vector<PendingLogitsContribution> logits_contributions;
    std::vector<std::pair<SlotState *, std::size_t>> prefix_cache_entries;

    int32_t effective_prefill_chunk_size = 0;
    SchedulerTickBudget tick_budget = {};

    double tick_time_ms = 0.0;
    double native_gpu_ms = 0.0;
    double native_sync_ms = 0.0;
    double native_logic_ms = 0.0;
    double total_tick_ms = 0.0;
  };

  std::vector<PendingLogitsContribution> scratch_logits_contributions_;
  std::vector<PendingLogitsContribution> pending_logits_contributions_;
  double total_decode_ms_ = 0.0;
  double total_prefill_ms_ = 0.0;
  std::size_t total_input_tokens_ = 0;
  std::size_t total_output_tokens_ = 0;
  std::size_t total_cache_hits_ = 0;
  std::size_t total_prefill_tokens_ = 0;
  PendingTickBookkeeping pending_bookkeeping_;
  bool has_pending_bookkeeping_ = false;

  mutable std::mutex request_queue_mutex_;
  mutable std::mutex operation_mutex_;
};

} // namespace noumena::cogentengine
