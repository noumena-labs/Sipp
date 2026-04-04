/////////////////////////////////////////////////////////////////////////////////////////////////
//
// inference_runtime.cpp
//
// - Inference-only runtime over llama.cpp.
// - Owns model lifetime, context reuse, and text generation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/inference_runtime.h"

#include <algorithm>
#include <chrono>

#include "runtime/llama/llama_utils.h"

namespace {

constexpr char kDefaultPromptContextKey[] = "__primary_prompt__";
constexpr int kMaxPredictionTokens = 2048;

noumena::cogentengine::InferenceRuntimeConfig
normalize_config(noumena::cogentengine::InferenceRuntimeConfig config) {
  config.n_seq_max = std::max<int32_t>(1, config.n_seq_max);
  config.gpu_layers = std::max<int32_t>(0, config.gpu_layers);
  config.max_cached_sessions = std::max<int32_t>(1, config.max_cached_sessions);
  config.retained_prefix_tokens =
      std::max<int32_t>(0, config.retained_prefix_tokens);
  config.prefill_chunk_size = std::max<int32_t>(0, config.prefill_chunk_size);
  config.scheduler_policy.decode_token_reserve =
      std::max<int32_t>(0, config.scheduler_policy.decode_token_reserve);
  return config;
}

} // namespace

namespace noumena::cogentengine {

bool InferenceRuntime::EnsureContextSpace(SequenceState &state,
                                          int new_tokens_needed, int n_ctx) {
  if (shared_context_ == nullptr || state.seq_id < 0 || n_ctx <= 0) {
    return false;
  }

  if (new_tokens_needed <= 0) {
    return true;
  }

  if (new_tokens_needed > n_ctx) {
    fprintf(stderr, "Input too large for context size!\n");
    return false;
  }

  if (state.n_past + new_tokens_needed <= n_ctx) {
    return true;
  }

  const int n_keep = std::min(config_.retained_prefix_tokens, state.n_past);
  const int required_discard = state.n_past + new_tokens_needed - n_ctx;
  const int max_discard = std::max(0, state.n_past - n_keep);
  const int n_discard = std::clamp(required_discard, 0, max_discard);

  llama_memory_t mem = llama_get_memory(shared_context_);

  if (n_discard <= 0) {
    if (!llama_memory_seq_rm(mem, state.seq_id, 0, -1)) {
      return false;
    }
    state.current_kv_tokens.clear();
    state.n_past = 0;
    return true;
  }

  if (!llama_memory_seq_rm(mem, state.seq_id, n_keep, n_keep + n_discard)) {
    return false;
  }

  llama_memory_seq_add(mem, state.seq_id, n_keep + n_discard, -1, -n_discard);

  if (static_cast<int>(state.current_kv_tokens.size()) > n_keep) {
    const int erase_end =
        std::min<int>(n_keep + n_discard, state.current_kv_tokens.size());
    const auto it_start = state.current_kv_tokens.begin() + n_keep;
    const auto it_end = state.current_kv_tokens.begin() + erase_end;
    state.current_kv_tokens.erase(it_start, it_end);
  } else {
    state.current_kv_tokens.clear();
  }

  state.n_past = std::max(0, state.n_past - n_discard);

  if (state.n_past + new_tokens_needed <= n_ctx) {
    return true;
  }

  if (!llama_memory_seq_rm(mem, state.seq_id, 0, -1)) {
    return false;
  }
  state.current_kv_tokens.clear();
  state.n_past = 0;

  return true;
}

bool InferenceRuntime::ExecutePromptTokensLocked(
    const std::string &context_key,
    const std::vector<llama_token> &prompt_tokens, int n_tokens_predict,
    TokenCallback on_token_received) {
  if (primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    return false;
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return false;
  }

  std::string model_context_key =
      context_key.empty() ? kDefaultPromptContextKey : context_key;
  if (prompt_tokens.empty()) {
    return true;
  }

  SequenceState *state = session_store_.Find(model_context_key);
  if (state == nullptr) {
    state = &session_store_.GetOrCreateSession(model_context_key);
  }

  session_store_.Touch(model_context_key);
  if (state == nullptr || state->seq_id < 0) {
    session_store_.Remove(model_context_key);
    return false;
  }

  const std::vector<llama_token> &new_tokens = prompt_tokens;
  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  llama_context *ctx = shared_context_;
  llama_memory_t mem = llama_get_memory(shared_context_);
  const bool is_recurrent = llama_model_is_recurrent(primary_model_);
  const bool is_hybrid = llama_model_is_hybrid(primary_model_);
  const bool allow_partial_kv = !(is_recurrent || is_hybrid);

  size_t match_len = 0;
  const size_t min_len =
      std::min(state->current_kv_tokens.size(), new_tokens.size());
  for (size_t i = 0; i < min_len; ++i) {
    if (state->current_kv_tokens[i] != new_tokens[i]) {
      break;
    }
    match_len++;
  }

  const int n_ctx = llama_n_ctx(ctx);
  const int tokens_to_add = static_cast<int>(new_tokens.size() - match_len);
  const int total_needed = tokens_to_add + n_tokens_predict;

  if (!EnsureContextSpace(*state, total_needed, n_ctx)) {
    return false;
  }

  if (match_len < state->current_kv_tokens.size()) {
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, state->seq_id, 0, -1);
      state->current_kv_tokens.clear();
      state->n_past = 0;
      match_len = 0;
    } else {
      if (!llama_memory_seq_rm(mem, state->seq_id,
                               static_cast<int32_t>(match_len), -1)) {
        fprintf(stderr, "failed to remove tokens from memory\n");
        return false;
      }
      state->current_kv_tokens.resize(match_len);
      state->n_past = static_cast<int>(match_len);
    }
  }

  llama_perf_context_reset(ctx);
  llama_sampler_reset(sampler_);
  llama_perf_sampler_reset(sampler_);
  const auto total_start = std::chrono::steady_clock::now();

  const int n_batch = static_cast<int>(llama_n_batch(ctx));
  llama_batch batch = llama_batch_init(
      n_batch, 0,
      static_cast<int32_t>(std::max<uint32_t>(1, llama_n_seq_max(ctx))));

  if (match_len == new_tokens.size() && match_len > 0) {
    if (!allow_partial_kv) {
      llama_memory_seq_rm(mem, state->seq_id, 0, -1);
      state->current_kv_tokens.clear();
      state->n_past = 0;
      match_len = 0;
    } else {
      if (!llama_memory_seq_rm(mem, state->seq_id,
                               static_cast<int32_t>(match_len - 1), -1)) {
        fprintf(stderr,
                "failed to remove last token from memory for re-evaluation\n");
        llama_batch_free(batch);
        return false;
      }
      state->current_kv_tokens.resize(match_len - 1);
      state->n_past = static_cast<int>(match_len - 1);
      match_len--;
    }
  }

  for (size_t i = match_len; i < new_tokens.size(); ++i) {
    const int batch_pos = static_cast<int>(i);
    const bool logits = (i == new_tokens.size() - 1);

    llama_utils::BatchAdd(batch, new_tokens[i], batch_pos, state->seq_id,
                          logits);

    if (batch.n_tokens >= n_batch) {
      if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s : failed to eval prompt\n", __func__);
        llama_batch_free(batch);
        return false;
      }
      state->n_past += batch.n_tokens;
      llama_utils::BatchClear(batch);
    }
  }

  if (batch.n_tokens > 0) {
    if (llama_decode(ctx, batch) != 0) {
      fprintf(stderr, "%s : failed to eval prompt final\n", __func__);
      llama_batch_free(batch);
      return false;
    }
    state->n_past += batch.n_tokens;
  }

  llama_synchronize(ctx);

  state->current_kv_tokens = new_tokens;

  llama_utils::BatchClear(batch);
  int output_token_count = 0;
  bool has_first_token_time = false;
  std::chrono::steady_clock::time_point first_token_time{};
  std::chrono::steady_clock::time_point last_token_time{};
  double accumulated_itl_ms = 0.0;
  double tail_itl_ms = 0.0;

  for (int i = 0; i < n_tokens_predict; ++i) {
    const llama_token tok = llama_sampler_sample(sampler_, ctx, -1);

    if (llama_vocab_is_eog(vocab, tok)) {
      break;
    }

    char buf[128];
    const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) {
      break;
    }
    output_token_count++;

    const auto token_time = std::chrono::steady_clock::now();
    if (!has_first_token_time) {
      first_token_time = token_time;
      has_first_token_time = true;
    } else {
      const double itl_ms =
          std::chrono::duration<double, std::milli>(token_time - last_token_time)
              .count();
      accumulated_itl_ms += itl_ms;
      tail_itl_ms = std::max(tail_itl_ms, itl_ms);
    }
    last_token_time = token_time;

    if (on_token_received) {
      on_token_received(buf, n);
    }

    llama_utils::BatchClear(batch);
    llama_utils::BatchAdd(batch, tok, state->n_past, state->seq_id, true);

    if (llama_decode(ctx, batch) != 0) {
      break;
    }

    llama_synchronize(ctx);

    state->n_past++;
    state->current_kv_tokens.push_back(tok);
  }

  llama_batch_free(batch);

  const auto total_end = std::chrono::steady_clock::now();
  const auto ctx_perf = llama_perf_context(ctx);
  const auto sampler_perf = llama_perf_sampler(sampler_);

  last_prompt_perf_ = PromptPerfStats{
      .total_ms =
          std::chrono::duration<double, std::milli>(total_end - total_start)
              .count(),
      .prompt_eval_ms = ctx_perf.t_p_eval_ms,
      .decode_eval_ms = ctx_perf.t_eval_ms,
      .sample_ms = sampler_perf.t_sample_ms,
      .queue_delay_ms = 0.0,
      .ttft_ms =
          has_first_token_time
              ? std::chrono::duration<double, std::milli>(first_token_time -
                                                          total_start)
                    .count()
              : 0.0,
      .mean_itl_ms =
          output_token_count > 1
              ? accumulated_itl_ms / static_cast<double>(output_token_count - 1)
              : 0.0,
      .tail_itl_ms = tail_itl_ms,
      .e2e_ms =
          std::chrono::duration<double, std::milli>(total_end - total_start)
              .count(),
      .input_token_count = static_cast<int32_t>(new_tokens.size()),
      .prompt_eval_tokens = ctx_perf.n_p_eval,
      .decode_eval_count = ctx_perf.n_eval,
      .sample_count = sampler_perf.n_sample,
      .output_token_count = output_token_count,
      .scheduler_tick_count = 0,
      .batch_participation_count = 0,
      .decode_first_tick_count = 0,
      .chunked_prefill_tick_count = 0,
      .mixed_workload_tick_count = 0,
  };
  has_last_prompt_perf_ = true;

  return true;
}

bool InferenceRuntime::ExecuteSingleSlotRequestLocked(SlotState &slot) {
  if (slot.request == nullptr || slot.session == nullptr) {
    slot.terminal_error_message = "Single-slot execution lost request state.";
    slot.phase = SlotPhase::Failed;
    return true;
  }

  GenerateRequest &request = *slot.request;
  request.lifecycle = GenerateRequestLifecycle::Running;
  request.emitted_token_count = 0;
  request.accumulated_itl_ms = 0.0;
  request.tail_itl_ms = 0.0;
  request.has_first_token_at = false;
  request.has_last_token_at = false;

  std::string output_text;
  const auto request_start = std::chrono::steady_clock::now();

  const bool success = ExecutePromptTokensLocked(
      request.context_key, request.prompt_tokens, request.max_output_tokens,
      [&](const char *token_piece, int32_t token_length) {
        const auto now = std::chrono::steady_clock::now();
        if (!request.has_first_token_at) {
          request.first_token_at = now;
          request.has_first_token_at = true;
        } else if (request.has_last_token_at) {
          const double itl_ms =
              std::chrono::duration<double, std::milli>(now -
                                                        request.last_token_at)
                  .count();
          request.accumulated_itl_ms += itl_ms;
          request.tail_itl_ms = std::max(request.tail_itl_ms, itl_ms);
        }

        request.last_token_at = now;
        request.has_last_token_at = true;
        request.emitted_token_count++;
        output_text.append(token_piece, static_cast<std::size_t>(token_length));

        if (request.on_token_received) {
          request.on_token_received(token_piece, token_length);
        }
      });

  slot.scheduler_tick_count++;
  slot.batch_participation_count++;
  slot.decode_step_count = static_cast<std::size_t>(last_prompt_perf_.decode_eval_count);
  slot.output_text = std::move(output_text);

  request.completed_at = std::chrono::steady_clock::now();
  request.has_completed_at = true;

  last_prompt_perf_.queue_delay_ms =
      request.has_admitted_at
          ? std::chrono::duration<double, std::milli>(request.admitted_at -
                                                      request.enqueued_at)
                .count()
          : 0.0;
  last_prompt_perf_.ttft_ms =
      request.has_first_token_at
          ? std::chrono::duration<double, std::milli>(request.first_token_at -
                                                      request.enqueued_at)
                .count()
          : 0.0;
  last_prompt_perf_.mean_itl_ms =
      request.emitted_token_count > 1
          ? request.accumulated_itl_ms /
                static_cast<double>(request.emitted_token_count - 1)
          : 0.0;
  last_prompt_perf_.tail_itl_ms = request.tail_itl_ms;
  last_prompt_perf_.e2e_ms =
      std::chrono::duration<double, std::milli>(request.completed_at -
                                                request.enqueued_at)
          .count();
  last_prompt_perf_.scheduler_tick_count =
      static_cast<int32_t>(slot.scheduler_tick_count);
  last_prompt_perf_.batch_participation_count =
      static_cast<int32_t>(slot.batch_participation_count);
  last_prompt_perf_.decode_first_tick_count = 0;
  last_prompt_perf_.chunked_prefill_tick_count = 0;
  last_prompt_perf_.mixed_workload_tick_count = 0;
  last_prompt_perf_.total_ms =
      std::max(last_prompt_perf_.total_ms,
               std::chrono::duration<double, std::milli>(request.completed_at -
                                                         request_start)
                   .count());

  if (!success) {
    slot.terminal_error_message = "Single-slot execution failed.";
    slot.phase = SlotPhase::Failed;
    request.lifecycle = GenerateRequestLifecycle::Failed;
    return true;
  }

  slot.phase = SlotPhase::Completed;
  request.lifecycle = GenerateRequestLifecycle::Completed;
  return true;
}

bool InferenceRuntime::RunPolicyBatchTickLocked() {
  if (primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    return false;
  }

  auto combine_slots = [](const std::vector<SlotState *> &left,
                          const std::vector<SlotState *> &right) {
    std::vector<SlotState *> combined;
    combined.reserve(left.size() + right.size());
    for (SlotState *slot : left) {
      if (slot != nullptr &&
          std::find(combined.begin(), combined.end(), slot) == combined.end()) {
        combined.push_back(slot);
      }
    }
    for (SlotState *slot : right) {
      if (slot != nullptr &&
          std::find(combined.begin(), combined.end(), slot) == combined.end()) {
        combined.push_back(slot);
      }
    }
    return combined;
  };

  std::vector<SlotState *> decode_ready_slots =
      slot_scheduler_.SelectDecodeReadySlots();
  std::vector<SlotState *> prefill_ready_slots =
      slot_scheduler_.SelectPrefillReadySlots();
  std::vector<SlotState *> runnable_slots =
      combine_slots(decode_ready_slots, prefill_ready_slots);
  if (runnable_slots.empty()) {
    return false;
  }

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  if (vocab == nullptr) {
    return false;
  }

  for (SlotState *slot : runnable_slots) {
    if (slot == nullptr || slot->request == nullptr || slot->session == nullptr ||
        slot->seq_id < 0) {
      if (slot != nullptr) {
        slot->terminal_error_message =
            "Runnable slot lost request or sequence state.";
        slot->phase = SlotPhase::Failed;
      }
      continue;
    }

    if (slot->sampler == nullptr) {
      slot->sampler = llama_sampler_clone(sampler_);
      if (slot->sampler == nullptr) {
        slot->terminal_error_message = "Failed to clone per-slot sampler.";
        slot->phase = SlotPhase::Failed;
        continue;
      }
    }

    GenerateRequest &request = *slot->request;
    SequenceState &session = *slot->session;

    if (slot->phase == SlotPhase::Prefill && slot->prefill_cursor == 0) {
      size_t match_len = 0;
      const size_t min_len =
          std::min(session.current_kv_tokens.size(), request.prompt_tokens.size());
      for (size_t i = 0; i < min_len; ++i) {
        if (session.current_kv_tokens[i] != request.prompt_tokens[i]) {
          break;
        }
        match_len++;
      }

      const int n_ctx = llama_n_ctx(shared_context_);
      const int tokens_to_add =
          static_cast<int>(request.prompt_tokens.size() - match_len);
      const int total_needed = tokens_to_add + request.max_output_tokens;
      if (!EnsureContextSpace(session, total_needed, n_ctx)) {
        slot->terminal_error_message = "Failed to ensure sequence context space.";
        slot->phase = SlotPhase::Failed;
        continue;
      }

      llama_memory_t mem = llama_get_memory(shared_context_);
      const bool is_recurrent = llama_model_is_recurrent(primary_model_);
      const bool is_hybrid = llama_model_is_hybrid(primary_model_);
      const bool allow_partial_kv = !(is_recurrent || is_hybrid);

      if (match_len < session.current_kv_tokens.size()) {
        if (!allow_partial_kv) {
          llama_memory_seq_rm(mem, session.seq_id, 0, -1);
          session.current_kv_tokens.clear();
          session.n_past = 0;
          match_len = 0;
        } else {
          if (!llama_memory_seq_rm(mem, session.seq_id,
                                   static_cast<int32_t>(match_len), -1)) {
            slot->terminal_error_message = "Failed to trim sequence memory.";
            slot->phase = SlotPhase::Failed;
            continue;
          }
          session.current_kv_tokens.resize(match_len);
          session.n_past = static_cast<int>(match_len);
        }
      }

      if (match_len == request.prompt_tokens.size() && match_len > 0) {
        if (!allow_partial_kv) {
          llama_memory_seq_rm(mem, session.seq_id, 0, -1);
          session.current_kv_tokens.clear();
          session.n_past = 0;
          match_len = 0;
        } else {
          if (!llama_memory_seq_rm(mem, session.seq_id,
                                   static_cast<int32_t>(match_len - 1), -1)) {
            slot->terminal_error_message =
                "Failed to re-evaluate last prompt token.";
            slot->phase = SlotPhase::Failed;
            continue;
          }
          session.current_kv_tokens.resize(match_len - 1);
          session.n_past = static_cast<int>(match_len - 1);
          match_len--;
        }
      }

      slot->prefill_cursor = match_len;
      slot->phase = slot->prefill_cursor >= request.prompt_tokens.size()
                        ? SlotPhase::Decode
                        : SlotPhase::Prefill;
    }

    request.lifecycle = GenerateRequestLifecycle::Running;
    llama_perf_sampler_reset(slot->sampler);
  }

  std::vector<SlotState *> live_decode_ready_slots =
      slot_scheduler_.SelectDecodeReadySlots();
  std::vector<SlotState *> live_prefill_ready_slots =
      slot_scheduler_.SelectPrefillReadySlots();
  std::vector<SlotState *> live_runnable_slots =
      combine_slots(live_decode_ready_slots, live_prefill_ready_slots);
  if (live_runnable_slots.empty()) {
    return false;
  }

  if (live_runnable_slots.size() == 1) {
    return ExecuteSingleSlotRequestLocked(*live_runnable_slots.front());
  }

  const int32_t batch_token_budget = config_.n_batch > 0 ? config_.n_batch : 256;
  const SchedulerTickBudget tick_budget = slot_scheduler_.BuildTickBudget(
      config_.scheduler_policy,
      static_cast<int32_t>(live_decode_ready_slots.size()),
      static_cast<int32_t>(live_prefill_ready_slots.size()), batch_token_budget,
      config_.prefill_chunk_size);
  SharedBatchPlan plan = batch_planner_.BuildPolicyBatch(
      live_decode_ready_slots, live_prefill_ready_slots, tick_budget,
      config_.prefill_chunk_size);
  if (plan.Empty()) {
    return false;
  }

  {
    std::vector<GenerateRequest *> tick_requests;
    tick_requests.reserve(plan.contributions.size());
    std::vector<GenerateRequest *> decode_requests;
    std::vector<GenerateRequest *> prefill_requests;

    const auto mark_request = [](std::vector<GenerateRequest *> &requests,
                                 GenerateRequest *request) {
      if (request == nullptr ||
          std::find(requests.begin(), requests.end(), request) != requests.end()) {
        return;
      }
      requests.push_back(request);
    };

    for (const BatchContribution &contribution : plan.contributions) {
      if (contribution.slot == nullptr || contribution.slot->request == nullptr) {
        continue;
      }
      mark_request(tick_requests, contribution.slot->request);
      if (contribution.kind == BatchContributionKind::Decode) {
        mark_request(decode_requests, contribution.slot->request);
      } else if (contribution.kind == BatchContributionKind::Prefill) {
        mark_request(prefill_requests, contribution.slot->request);
      }
    }

    if (plan.prefill_token_count > 0 && plan.decode_token_count > 0) {
      for (GenerateRequest *request : tick_requests) {
        request->mixed_workload_tick_count++;
      }
    }
    if (tick_budget.EffectiveDecodeBudget() > 0) {
      for (GenerateRequest *request : decode_requests) {
        request->decode_first_tick_count++;
      }
    }
    if (config_.prefill_chunk_size > 0) {
      for (GenerateRequest *request : prefill_requests) {
        request->chunked_prefill_tick_count++;
      }
    }
  }

  shared_batch_builder_.EnsureCapacity(batch_token_budget,
                                       std::max<int32_t>(1, config_.n_seq_max));
  shared_batch_builder_.Reset();

  std::vector<const BatchContribution *> logits_contributions;
  logits_contributions.reserve(plan.contributions.size());

  for (const BatchContribution &contribution : plan.contributions) {
    if (contribution.slot == nullptr || contribution.slot->seq_id < 0) {
      continue;
    }

    const bool added = contribution.kind == BatchContributionKind::Prefill
                           ? shared_batch_builder_.AddPrefillToken(
                                 contribution.token, contribution.position,
                                 contribution.slot->seq_id,
                                 contribution.request_logits)
                           : shared_batch_builder_.AddDecodeToken(
                                 contribution.token, contribution.position,
                                 contribution.slot->seq_id,
                                 contribution.request_logits);
    if (!added) {
      if (contribution.slot != nullptr) {
        contribution.slot->terminal_error_message =
            "Shared batch builder capacity was exceeded.";
        contribution.slot->phase = SlotPhase::Failed;
      }
      return false;
    }

    if (contribution.request_logits) {
      logits_contributions.push_back(&contribution);
    }
  }

  llama_perf_context_reset(shared_context_);
  const auto tick_start = std::chrono::steady_clock::now();

  if (llama_decode(shared_context_, shared_batch_builder_.Get()) != 0) {
    for (SlotState *slot : live_runnable_slots) {
      if (slot == nullptr) {
        continue;
      }
      slot->terminal_error_message = "llama_decode() failed in shared tick.";
      slot->phase = SlotPhase::Failed;
      if (slot->request != nullptr) {
        slot->request->lifecycle = GenerateRequestLifecycle::Failed;
      }
    }
    return false;
  }

  llama_synchronize(shared_context_);

  for (const BatchContribution &contribution : plan.contributions) {
    if (contribution.slot == nullptr || contribution.slot->session == nullptr) {
      continue;
    }

    SequenceState &session = *contribution.slot->session;
    session.current_kv_tokens.push_back(contribution.token);
    session.n_past++;
  }

  batch_planner_.ApplyDecodeResults(plan);

  int32_t logit_index = 0;
  for (const BatchContribution *logit_contribution : logits_contributions) {
    if (logit_contribution == nullptr || logit_contribution->slot == nullptr ||
        logit_contribution->slot->request == nullptr ||
        logit_contribution->slot->sampler == nullptr) {
      logit_index++;
      continue;
    }

    SlotState &slot = *logit_contribution->slot;
    GenerateRequest &slot_request = *slot.request;
    const llama_token next_token =
        llama_sampler_sample(slot.sampler, shared_context_, logit_index++);

    if (llama_vocab_is_eog(vocab, next_token)) {
      slot.phase = SlotPhase::Completed;
      slot_request.lifecycle = GenerateRequestLifecycle::Completed;
      continue;
    }

    char piece_buffer[128];
    const int piece_length = llama_token_to_piece(
        vocab, next_token, piece_buffer, sizeof(piece_buffer), 0, true);
    if (piece_length < 0) {
      slot.terminal_error_message =
          "Failed to convert sampled token to text piece.";
      slot.phase = SlotPhase::Failed;
      slot_request.lifecycle = GenerateRequestLifecycle::Failed;
      continue;
    }

    slot.generated_tokens.push_back(next_token);
    slot.buffered_output_text.append(piece_buffer,
                                     static_cast<std::size_t>(piece_length));
    slot.phase = SlotPhase::Streaming;
    slot_request.lifecycle = GenerateRequestLifecycle::Streaming;
    slot_scheduler_.EmitBufferedTokenPiece(slot);

    if (slot_request.max_output_tokens > 0 &&
        static_cast<int32_t>(slot.generated_tokens.size()) >=
            slot_request.max_output_tokens) {
      slot.phase = SlotPhase::Completed;
      slot_request.lifecycle = GenerateRequestLifecycle::Completed;
    } else if (slot.phase != SlotPhase::Failed) {
      slot.phase = SlotPhase::Decode;
      slot_request.lifecycle = GenerateRequestLifecycle::Running;
    }
  }

  const auto tick_end = std::chrono::steady_clock::now();
  const auto ctx_perf = llama_perf_context(shared_context_);

  if (!has_last_prompt_perf_) {
    last_prompt_perf_ = {};
    for (SlotState *slot : live_runnable_slots) {
      if (slot != nullptr && slot->request != nullptr) {
        last_prompt_perf_.input_token_count +=
            static_cast<int32_t>(slot->request->prompt_tokens.size());
      }
    }
    has_last_prompt_perf_ = true;
  }

  last_prompt_perf_.total_ms +=
      std::chrono::duration<double, std::milli>(tick_end - tick_start).count();
  last_prompt_perf_.prompt_eval_ms += ctx_perf.t_p_eval_ms;
  last_prompt_perf_.decode_eval_ms += ctx_perf.t_eval_ms;
  last_prompt_perf_.prompt_eval_tokens += plan.prefill_token_count;
  last_prompt_perf_.decode_eval_count += plan.decode_token_count;
  last_prompt_perf_.sample_count +=
      static_cast<int32_t>(logits_contributions.size());
  double tick_sample_ms = 0.0;
  last_prompt_perf_.output_token_count = 0;
  for (SlotState *slot : live_runnable_slots) {
    if (slot != nullptr && slot->sampler != nullptr) {
      tick_sample_ms += llama_perf_sampler(slot->sampler).t_sample_ms;
      last_prompt_perf_.output_token_count +=
          static_cast<int32_t>(slot->generated_tokens.size());
    }
  }
  last_prompt_perf_.sample_ms += tick_sample_ms;

  UpdateSharedBatchMetricsLocked(plan);
  UpdateSchedulerPerfCountersLocked(plan, tick_budget);
  return true;
}

bool InferenceRuntime::RunSharedBatchTickLocked() {
  return RunPolicyBatchTickLocked();
}

void InferenceRuntime::UpdateSharedBatchMetricsLocked(
    const SharedBatchPlan &plan) {
  if (plan.Empty()) {
    return;
  }

  shared_batch_stats_.tick_count++;
  shared_batch_stats_.total_occupied_slots +=
      static_cast<std::uint64_t>(std::max(0, plan.occupied_slot_count));
  shared_batch_stats_.total_prefill_tokens +=
      static_cast<std::uint64_t>(std::max(0, plan.prefill_token_count));
  shared_batch_stats_.total_decode_tokens +=
      static_cast<std::uint64_t>(std::max(0, plan.decode_token_count));
}

void InferenceRuntime::UpdateSchedulerPerfCountersLocked(
    const SharedBatchPlan &plan, const SchedulerTickBudget &budget) {
  // Phase 4 algorithm steps:
  // 1. Record whether this tick used explicit decode reservation.
  // 2. Record whether chunked prefill was active.
  // 3. Record whether the tick mixed decode and prefill contributions.
  // 4. Later, attach real queue delay, TTFT, ITL, and tail ITL once the
  //    request lifecycle carries precise timestamps.
  scheduler_perf_counters_.tick_count++;
  if (budget.EffectiveDecodeBudget() > 0 && plan.decode_token_count > 0) {
    scheduler_perf_counters_.decode_first_tick_count++;
  }
  if (config_.prefill_chunk_size > 0 && plan.prefill_token_count > 0) {
    scheduler_perf_counters_.chunked_prefill_tick_count++;
  }
  if (plan.decode_token_count > 0 && plan.prefill_token_count > 0) {
    scheduler_perf_counters_.mixed_workload_tick_count++;
  }
}

InferenceRuntime::InferenceRuntime(std::string model_path,
                                   InferenceRuntimeConfig config)
    : config_(normalize_config(config)),
      session_store_(static_cast<size_t>(config_.max_cached_sessions),
                     static_cast<size_t>(std::max<int32_t>(1, config_.n_seq_max))) {
  if (model_path.empty()) {
    fprintf(stderr, "%s: error: model path is required\n", __func__);
    return;
  }

#if defined(NDEBUG) || defined(CE_SUPPRESS_LLAMA_LOGS)
  llama_log_set(llama_utils::LogCallbackDefault, nullptr);
#endif

  ggml_backend_load_all();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = config_.gpu_layers;
  model_params.use_mlock = false;
#if defined(__EMSCRIPTEN__)
  model_params.use_mmap = false;
#else
  model_params.use_mmap = true;
#endif

  ggml_backend_dev_t cpu_only_devices[2] = {nullptr, nullptr};
  if (config_.gpu_layers == 0) {
    cpu_only_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_only_devices[0] != nullptr) {
      model_params.devices = cpu_only_devices;
    }
  }

  primary_model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (primary_model_ == nullptr) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return;
  }

  shared_context_ = CreateContext();
  if (shared_context_ == nullptr) {
    fprintf(stderr, "%s: error: failed to create shared context\n", __func__);
    return;
  }
  session_store_.BindSharedContext(shared_context_);

  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = false;
  sampler_ = llama_sampler_chain_init(sparams);

  llama_sampler_chain_add(sampler_,
                          llama_sampler_init_penalties(64, 1.05f, 0.0f, 0.0f));
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.8f, 1));
  llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(sampler_,
                          llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  slot_scheduler_.Resize(
      static_cast<std::size_t>(std::max<int32_t>(1, config_.n_seq_max)));
  shared_batch_builder_.EnsureCapacity(config_.n_batch > 0 ? config_.n_batch
                                                           : 256,
                                       std::max<int32_t>(1, config_.n_seq_max));
}

llama_context *InferenceRuntime::CreateContext() const {
  if (primary_model_ == nullptr) {
    return nullptr;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx =
      config_.n_ctx > 0
          ? static_cast<uint32_t>(config_.n_ctx)
          : static_cast<uint32_t>(
                std::min(4096 * 2, llama_model_n_ctx_train(primary_model_)));
  ctx_params.n_batch =
      config_.n_batch > 0 ? static_cast<uint32_t>(config_.n_batch) : 256u;
  if (config_.n_ubatch > 0) {
    ctx_params.n_ubatch = static_cast<uint32_t>(config_.n_ubatch);
  } else if (ctx_params.n_ubatch > ctx_params.n_batch) {
    ctx_params.n_ubatch = ctx_params.n_batch;
  }
  ctx_params.n_seq_max = static_cast<uint32_t>(config_.n_seq_max);
  ctx_params.n_threads = config_.n_threads > 0
                             ? config_.n_threads
                             : llama_utils::DefaultThreadCount();
  ctx_params.n_threads_batch = config_.n_threads_batch > 0
                                   ? config_.n_threads_batch
                                   : ctx_params.n_threads;
  ctx_params.no_perf = false;

  if (config_.flash_attention >= 0) {
    ctx_params.flash_attn_type =
        static_cast<llama_flash_attn_type>(config_.flash_attention);
  }
  if (config_.kv_unified >= 0) {
    ctx_params.kv_unified = config_.kv_unified != 0;
  }

  return llama_init_from_model(primary_model_, ctx_params);
}

InferenceRuntime::~InferenceRuntime() {
  if (sampler_ != nullptr) {
    llama_sampler_free(sampler_);
  }

  session_store_.Clear();

  if (shared_context_ != nullptr) {
    llama_free(shared_context_);
  }

  if (primary_model_ != nullptr) {
    llama_model_free(primary_model_);
  }

  llama_backend_free();
}

bool InferenceRuntime::IsReady() const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  return primary_model_ != nullptr && shared_context_ != nullptr &&
         sampler_ != nullptr;
}

bool InferenceRuntime::TryGetLastPromptPerf(PromptPerfStats &out) const {
  std::lock_guard<std::mutex> lock(operation_mutex_);
  if (!has_last_prompt_perf_) {
    return false;
  }

  out = last_prompt_perf_;
  return true;
}

GenerateRequestId
InferenceRuntime::EnqueueRequest(std::string context_key, std::string prompt,
                                 int n_tokens_predict,
                                 TokenCallback on_token_received) {
  std::lock_guard<std::mutex> lock(operation_mutex_);

  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return 0;
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return 0;
  }
  if (context_key.empty()) {
    context_key = kDefaultPromptContextKey;
  }

  GenerateRequest request;
  request.id = next_request_id_++;
  request.context_key = std::move(context_key);
  request.prompt_text = std::move(prompt);
  request.max_output_tokens = n_tokens_predict;
  request.on_token_received = std::move(on_token_received);

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  request.prompt_tokens =
      llama_utils::Tokenize(vocab, request.prompt_text, false, true);

  if (!request_queue_.Push(std::move(request))) {
    return 0;
  }

  return next_request_id_ - 1;
}

bool InferenceRuntime::RunUntilRequestCompletes(
    GenerateRequestId request_id, GenerateResponse &out_response) {
  std::lock_guard<std::mutex> lock(operation_mutex_);

  out_response = {};
  has_last_prompt_perf_ = false;

  const auto commit_completed_perf = [this](const GenerateResponse &response) {
    const PromptPerfStats accumulated_perf = last_prompt_perf_;
    last_prompt_perf_ = response.perf;
    last_prompt_perf_.total_ms = accumulated_perf.total_ms > 0.0
                                     ? accumulated_perf.total_ms
                                     : std::max(response.perf.total_ms,
                                                response.perf.e2e_ms);
    last_prompt_perf_.prompt_eval_ms = accumulated_perf.prompt_eval_ms;
    last_prompt_perf_.decode_eval_ms = accumulated_perf.decode_eval_ms;
    last_prompt_perf_.sample_ms = accumulated_perf.sample_ms;
    last_prompt_perf_.prompt_eval_tokens = accumulated_perf.prompt_eval_tokens;
    last_prompt_perf_.decode_eval_count =
        std::max(last_prompt_perf_.decode_eval_count,
                 accumulated_perf.decode_eval_count);
    last_prompt_perf_.sample_count =
        std::max(last_prompt_perf_.sample_count, accumulated_perf.sample_count);
    last_prompt_perf_.output_token_count =
        std::max(last_prompt_perf_.output_token_count,
                 accumulated_perf.output_token_count);
    has_last_prompt_perf_ = true;

    scheduler_perf_counters_.accumulated_queue_delay_ms +=
        response.perf.queue_delay_ms;
    scheduler_perf_counters_.accumulated_ttft_ms += response.perf.ttft_ms;
    scheduler_perf_counters_.max_tail_itl_ms =
        std::max(scheduler_perf_counters_.max_tail_itl_ms,
                 response.perf.tail_itl_ms);
  };

  if (request_id == 0 || primary_model_ == nullptr || shared_context_ == nullptr ||
      sampler_ == nullptr) {
    return false;
  }

  while (true) {
    if (auto completed = request_queue_.TakeCompletedResponse(request_id);
        completed.has_value()) {
      out_response = std::move(*completed);
      commit_completed_perf(out_response);
      return out_response.status == GenerateResponseStatus::Completed;
    }

    GenerateRequest *target_request = request_queue_.FindMutable(request_id);
    if (target_request == nullptr) {
      return false;
    }

    while (
        slot_scheduler_.AdmitPendingRequests(request_queue_, session_store_)) {
    }

    if (auto completed = request_queue_.TakeCompletedResponse(request_id);
        completed.has_value()) {
      out_response = std::move(*completed);
      commit_completed_perf(out_response);
      return out_response.status == GenerateResponseStatus::Completed;
    }

    const bool tick_executed = RunPolicyBatchTickLocked();
    if (!tick_executed) {
      SlotState *active_slot = slot_scheduler_.FindFirstActiveSlot();
      if (active_slot == nullptr) {
        return false;
      }
      if (active_slot->phase != SlotPhase::Failed &&
          active_slot->phase != SlotPhase::Completed) {
        active_slot->terminal_error_message =
            "Shared batch tick could not make progress.";
        active_slot->phase = SlotPhase::Failed;
      }
    }

    slot_scheduler_.FinalizeCompletedSlots(request_queue_, session_store_);
  }
}

bool InferenceRuntime::Prompt(std::string model_context_key, std::string prompt,
                              int n_tokens_predict,
                              TokenCallback on_token_received) {
  // Phase 2 note:
  // - This is still the live Phase 1 execution path.
  // - Once request queue ownership is implemented, keep Prompt(...) as the
  //   synchronous wrapper that enqueues a request and waits for completion.
  std::lock_guard<std::mutex> lock(operation_mutex_);
  has_last_prompt_perf_ = false;

  if (primary_model_ == nullptr || sampler_ == nullptr) {
    return false;
  }
  if (n_tokens_predict <= 0 || n_tokens_predict > kMaxPredictionTokens) {
    return false;
  }
  if (model_context_key.empty()) {
    model_context_key = kDefaultPromptContextKey;
  }

  const llama_vocab *vocab = llama_model_get_vocab(primary_model_);
  std::vector<llama_token> new_tokens =
      llama_utils::Tokenize(vocab, prompt, false, true);
  return ExecutePromptTokensLocked(model_context_key, new_tokens,
                                   n_tokens_predict, on_token_received);
}

} // namespace noumena::cogentengine
