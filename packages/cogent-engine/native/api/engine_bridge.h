#pragma once

#include <string>

#include "ffi_types.h"

int CE_InitPlugin(const char* model_path);
void CE_ClosePlugin();
int CE_GetLastPromptPerf(CE_PromptPerfMetrics* out_metrics);
std::string CE_ProcessPromptQuery(
    const char* context_key,
    const char* prompt,
    int n_tokens_predict);
