#include <emscripten/emscripten.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

#include "engine_bridge.h"

namespace {

constexpr int kStatusFailure = -1;
constexpr int kStatusInvalidArguments = -2;
constexpr int kStatusNotInitialized = -3;
constexpr int kMaxPromptTokens = 2048;

bool is_valid_prediction_tokens(int token_count) {
  return token_count > 0 && token_count <= kMaxPromptTokens;
}

bool g_isEngineInitialized = false;
std::mutex g_apiMutex;

char *duplicate_heap_string(const char *value) {
  const char *source = value != nullptr ? value : "";
  const std::size_t length = std::strlen(source);
  char *out = static_cast<char *>(std::malloc(length + 1));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, source, length + 1);
  return out;
}

int init_engine_locked(const char *model_path, int n_ctx, int n_batch,
                       int n_ubatch, int n_seq_max, int n_threads,
                       int n_threads_batch, int gpu_layers,
                       int flash_attention, int kv_unified,
                       int max_cached_sessions, int retained_prefix_tokens,
                       int prefill_chunk_size,
                       int prefix_cache_interval_tokens,
                       int max_prefix_cache_entries, int scheduler_policy,
                       int decode_token_reserve,
                       int adaptive_prefill_chunking,
                       int enable_runtime_observability,
                       int enable_backend_profiling,
                       const char *mmproj_path, int multimodal_use_gpu,
                       int debug_compare_multimodal_embeddings,
                       int image_min_tokens,
                       int image_max_tokens, int sampling_repeat_last_n,
                       float sampling_repeat_penalty,
                       float sampling_frequency_penalty,
                       float sampling_presence_penalty, int sampling_top_k,
                       float sampling_top_p, float sampling_min_p,
                       float sampling_temperature, int sampling_seed) {
  if (!model_path || std::strlen(model_path) == 0) {
    return kStatusInvalidArguments;
  }

  if (g_isEngineInitialized) {
    CE_ClosePlugin();
    g_isEngineInitialized = false;
  }

  const CE_InitConfig config{
      .n_ctx = n_ctx,
      .n_batch = n_batch,
      .n_ubatch = n_ubatch,
      .n_seq_max = n_seq_max,
      .n_threads = n_threads,
      .n_threads_batch = n_threads_batch,
      .gpu_layers = gpu_layers,
      .flash_attention = flash_attention,
      .kv_unified = kv_unified,
      .max_cached_sessions = max_cached_sessions,
      .retained_prefix_tokens = retained_prefix_tokens,
      .prefill_chunk_size = prefill_chunk_size,
      .prefix_cache_interval_tokens = prefix_cache_interval_tokens,
      .max_prefix_cache_entries = max_prefix_cache_entries,
      .scheduler_policy = scheduler_policy,
      .decode_token_reserve = decode_token_reserve,
      .adaptive_prefill_chunking = adaptive_prefill_chunking,
      .enable_runtime_observability = enable_runtime_observability,
      .enable_backend_profiling = enable_backend_profiling,
      .mmproj_path = mmproj_path,
      .multimodal_use_gpu = multimodal_use_gpu,
      .debug_compare_multimodal_embeddings =
          debug_compare_multimodal_embeddings,
      .image_min_tokens = image_min_tokens,
      .image_max_tokens = image_max_tokens,
      .sampling_repeat_last_n = sampling_repeat_last_n,
      .sampling_repeat_penalty = sampling_repeat_penalty,
      .sampling_frequency_penalty = sampling_frequency_penalty,
      .sampling_presence_penalty = sampling_presence_penalty,
      .sampling_top_k = sampling_top_k,
      .sampling_top_p = sampling_top_p,
      .sampling_min_p = sampling_min_p,
      .sampling_temperature = sampling_temperature,
      .sampling_seed = sampling_seed,
  };

  const int init_status = CE_InitPlugin(model_path, &config);
  if (init_status != 0) {
    return init_status;
  }

  g_isEngineInitialized = true;
  return 0;
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int CE_Init(const char *model_path, int n_ctx, int n_batch, int n_ubatch,
            int n_seq_max, int n_threads, int n_threads_batch, int gpu_layers,
            int flash_attention, int kv_unified, int max_cached_sessions,
            int retained_prefix_tokens, int prefill_chunk_size,
            int prefix_cache_interval_tokens, int max_prefix_cache_entries,
            int scheduler_policy, int decode_token_reserve,
            int adaptive_prefill_chunking, int enable_runtime_observability,
            int enable_backend_profiling, int multimodal_use_gpu,
            int debug_compare_multimodal_embeddings,
            int sampling_repeat_last_n,
            float sampling_repeat_penalty, float sampling_frequency_penalty,
            float sampling_presence_penalty, int sampling_top_k,
            float sampling_top_p, float sampling_min_p,
            float sampling_temperature, int sampling_seed) {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  return init_engine_locked(
      model_path, n_ctx, n_batch, n_ubatch, n_seq_max, n_threads,
      n_threads_batch, gpu_layers, flash_attention, kv_unified,
      max_cached_sessions, retained_prefix_tokens, prefill_chunk_size,
      prefix_cache_interval_tokens, max_prefix_cache_entries,
      scheduler_policy, decode_token_reserve, adaptive_prefill_chunking,
      enable_runtime_observability, enable_backend_profiling, nullptr,
      multimodal_use_gpu, debug_compare_multimodal_embeddings, 0, 0,
      sampling_repeat_last_n, sampling_repeat_penalty,
      sampling_frequency_penalty, sampling_presence_penalty, sampling_top_k,
      sampling_top_p, sampling_min_p, sampling_temperature, sampling_seed);
}

EMSCRIPTEN_KEEPALIVE
int CE_InitWithMultimodal(
    const char *model_path, int n_ctx, int n_batch, int n_ubatch,
    int n_seq_max, int n_threads, int n_threads_batch, int gpu_layers,
    int flash_attention, int kv_unified, int max_cached_sessions,
    int retained_prefix_tokens, int prefill_chunk_size,
    int prefix_cache_interval_tokens, int max_prefix_cache_entries,
    int scheduler_policy, int decode_token_reserve,
    int adaptive_prefill_chunking, int enable_runtime_observability,
    int enable_backend_profiling, const char *mmproj_path,
    int multimodal_use_gpu, int debug_compare_multimodal_embeddings,
    int image_min_tokens, int image_max_tokens, int sampling_repeat_last_n,
    float sampling_repeat_penalty, float sampling_frequency_penalty,
    float sampling_presence_penalty, int sampling_top_k,
    float sampling_top_p, float sampling_min_p,
    float sampling_temperature, int sampling_seed) {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  return init_engine_locked(
      model_path, n_ctx, n_batch, n_ubatch, n_seq_max, n_threads,
      n_threads_batch, gpu_layers, flash_attention, kv_unified,
      max_cached_sessions, retained_prefix_tokens, prefill_chunk_size,
      prefix_cache_interval_tokens, max_prefix_cache_entries,
      scheduler_policy, decode_token_reserve, adaptive_prefill_chunking,
      enable_runtime_observability, enable_backend_profiling, mmproj_path,
      multimodal_use_gpu, debug_compare_multimodal_embeddings,
      image_min_tokens, image_max_tokens, sampling_repeat_last_n,
      sampling_repeat_penalty, sampling_frequency_penalty,
      sampling_presence_penalty, sampling_top_k, sampling_top_p,
      sampling_min_p, sampling_temperature, sampling_seed);
}

EMSCRIPTEN_KEEPALIVE
void CE_Close() {
  std::lock_guard<std::mutex> lock(g_apiMutex);

  if (!g_isEngineInitialized) {
    return;
  }

  CE_ClosePlugin();
  g_isEngineInitialized = false;
}


EMSCRIPTEN_KEEPALIVE
char *CE_GetBackendObservabilityJson() {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  return duplicate_heap_string(CE_GetBackendObservabilityJsonString());
}

EMSCRIPTEN_KEEPALIVE
const char *CE_GetMediaMarker() {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  if (!g_isEngineInitialized) {
    return nullptr;
  }
  return CE_GetMediaMarkerString();
}

EMSCRIPTEN_KEEPALIVE
const char *CE_GetChatTemplate() {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  if (!g_isEngineInitialized) {
    return nullptr;
  }
  return CE_GetChatTemplateString();
}

EMSCRIPTEN_KEEPALIVE
char *CE_ApplyChatTemplate(const char *messages_json, int add_assistant) {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  if (!g_isEngineInitialized) {
    return duplicate_heap_string("");
  }
  return duplicate_heap_string(
      CE_ApplyChatTemplateString(messages_json, add_assistant));
}

EMSCRIPTEN_KEEPALIVE
CE_RequestId CE_EnqueuePrompt(const char *context_key, const char *prompt,
                              int n_tokens, CE_TokenCallback on_token) {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  if (!g_isEngineInitialized) {
    return 0;
  }
  if (prompt == nullptr || !is_valid_prediction_tokens(n_tokens)) {
    return 0;
  }

  return CE_EnqueuePromptQuery(context_key, prompt, n_tokens, on_token);
}

EMSCRIPTEN_KEEPALIVE
CE_RequestId CE_EnqueuePromptWithMedia(
    const char *context_key, const char *prompt, int n_tokens, int n_images,
    const uint8_t *images_flat_buffer, const int32_t *image_sizes,
    CE_TokenCallback on_token) {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  if (!g_isEngineInitialized) {
    return 0;
  }
  if (prompt == nullptr || !is_valid_prediction_tokens(n_tokens)) {
    return 0;
  }

  return CE_EnqueuePromptWithMediaQuery(context_key, prompt, n_tokens,
                                        n_images, images_flat_buffer,
                                        image_sizes, on_token);
}


EMSCRIPTEN_KEEPALIVE
int CE_CancelQueuedRequest(CE_RequestId request_id) {
  std::lock_guard<std::mutex> lock(g_apiMutex);
  if (!g_isEngineInitialized || request_id == 0) {
    return 0;
  }
  return CE_CancelQueuedPromptQuery(request_id);
}

EMSCRIPTEN_KEEPALIVE
void CE_FreeString(char *str) {
  if (str) {
    std::free(str);
  }
}

} // extern "C"
