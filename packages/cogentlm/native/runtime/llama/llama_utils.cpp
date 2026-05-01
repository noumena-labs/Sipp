/////////////////////////////////////////////////////////////////////////////////////////////////
//
// llama_utils.cpp
//
// - Shared llama.cpp helpers used by the inference runtime.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "runtime/llama/llama_utils.h"

#include <algorithm>
#include <thread>

namespace noumena::cogentengine::llama_utils {

std::vector<llama_token> Tokenize(
    const struct llama_vocab* vocab,
    const std::string& text,
    bool add_special,
    bool parse_special) {
  int n_tokens = static_cast<int>(text.length()) + 2 * static_cast<int>(add_special);
  std::vector<llama_token> result(n_tokens);
  n_tokens = llama_tokenize(
      vocab,
      text.data(),
      text.length(),
      result.data(),
      result.size(),
      add_special,
      parse_special);
  if (n_tokens < 0) {
    result.resize(-n_tokens);
    const int check = llama_tokenize(
        vocab,
        text.data(),
        text.length(),
        result.data(),
        result.size(),
        add_special,
        parse_special);
    GGML_ASSERT(check == -n_tokens);
  } else {
    result.resize(n_tokens);
  }
  return result;
}

void BatchClear(struct llama_batch& batch) {
  batch.n_tokens = 0;
}

void BatchAdd(
    struct llama_batch& batch,
    llama_token id,
    llama_pos pos,
    llama_seq_id seq_id,
    bool logits) {
  GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

  batch.token[batch.n_tokens] = id;
  batch.pos[batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = 1;
  batch.seq_id[batch.n_tokens][0] = seq_id;
  batch.logits[batch.n_tokens] = logits;
  batch.n_tokens++;
}

void LogCallbackDefault(enum ggml_log_level level, const char* text, void* user_data) {
  (void) text;
  (void) level;
  (void) user_data;
}

int DefaultThreadCount() {
#if defined(__EMSCRIPTEN__)
  #if defined(GGML_PTHREADS) && GGML_PTHREADS
    #if defined(CE_WASM_PTHREAD_POOL_SIZE)
      return std::clamp(CE_WASM_PTHREAD_POOL_SIZE, 1, 8);
    #else
      return 2;
    #endif
  #else
    return 1;
  #endif
#else
  return std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 8);
#endif
}

}  // namespace noumena::cogentengine::llama_utils
