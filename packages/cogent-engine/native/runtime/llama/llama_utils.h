/////////////////////////////////////////////////////////////////////////////////////////////////
//
// llama_utils.h
//
// - Shared llama.cpp helpers used by the inference runtime.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include <vector>

#include "llama.h"

namespace noumena::cogentengine::llama_utils {

std::vector<llama_token> Tokenize(
    const struct llama_vocab* vocab,
    const std::string& text,
    bool add_special,
    bool parse_special);

void BatchClear(struct llama_batch& batch);

void BatchAdd(
    struct llama_batch& batch,
    llama_token id,
    llama_pos pos,
    const std::vector<llama_seq_id>& seq_ids,
    bool logits);

void LogCallbackDefault(enum ggml_log_level level, const char* text, void* user_data);

int DefaultThreadCount();

}  // namespace noumena::cogentengine::llama_utils
