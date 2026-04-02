#pragma once

#include <string>

#include "ffi_types.h"

int CE_InitPlugin(const char* model_path, const CE_InitConfig* config);
void CE_ClosePlugin();
int CE_GetLastPromptPerf(CE_PromptPerfMetrics* out_metrics);
int CE_StreamPromptQuery(
    const char* context_key,
    const char* prompt,
    int n_tokens_predict,
    CE_TokenCallback on_token);
