/////////////////////////////////////////////////////////////////////////////////////////////////
//
// inference_runtime.h
//
// - Inference-only runtime over llama.cpp.
// - Owns model lifetime, context reuse, and text generation.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(__CYGWIN32__)
#define COGENTENGINE_INTERFACE_API __stdcall
#define COGENTENGINE_INTERFACE_EXPORT __declspec(dllexport)
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(_WIN64) || defined(WINAPI_FAMILY)
#define COGENTENGINE_INTERFACE_API __stdcall
#define COGENTENGINE_INTERFACE_EXPORT __declspec(dllexport)
#elif defined(__MACH__) || defined(__ANDROID__) || defined(__linux__) || defined(LUMIN)
#define COGENTENGINE_INTERFACE_API
#define COGENTENGINE_INTERFACE_EXPORT __attribute__ ((visibility ("default")))
#else
#define COGENTENGINE_INTERFACE_API
#define COGENTENGINE_INTERFACE_EXPORT
#endif

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

#include "runtime/session/session_store.h"

struct llama_model;
struct llama_sampler;

namespace noumena::cogentengine {

struct PromptPerfStats {
  double total_ms = 0.0;
  double prompt_eval_ms = 0.0;
  double decode_eval_ms = 0.0;
  double sample_ms = 0.0;
  int32_t prompt_eval_tokens = 0;
  int32_t decode_eval_count = 0;
  int32_t sample_count = 0;
  int32_t output_token_count = 0;
};

class COGENTENGINE_INTERFACE_EXPORT InferenceRuntime {
public:
  explicit InferenceRuntime(std::string model_path, int gpu_layers_n = 99);
  ~InferenceRuntime();

  bool IsReady() const;
  bool TryGetLastPromptPerf(PromptPerfStats& out) const;

  std::string Prompt(
      std::string context_key,
      std::string prompt,
      int n_tokens_predict = 64,
      std::function<void(std::string)> onTokenReceived = nullptr);

private:
  bool EnsureContextSpace(ContextState& state, int new_tokens_needed, int n_ctx);

  llama_model* primary_model_ = nullptr;
  llama_sampler* sampler_ = nullptr;
  PromptPerfStats last_prompt_perf_;
  bool has_last_prompt_perf_ = false;
  SessionStore session_store_;
  mutable std::mutex operation_mutex_;
};

}  // namespace noumena::cogentengine
