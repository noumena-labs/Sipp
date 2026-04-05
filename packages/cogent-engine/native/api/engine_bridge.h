#pragma once

#include <string>

#include "ffi_types.h"

int CE_InitPlugin(const char* model_path, const CE_InitConfig* config);
void CE_ClosePlugin();
int CE_GetLastPromptPerf(CE_PromptPerfMetrics* out_metrics);
const char* CE_GetBackendInfoJsonString();
CE_RequestId CE_EnqueuePromptQuery(
    const char* context_key,
    const char* prompt,
    int n_tokens_predict,
    CE_TokenCallback on_token);
int CE_CancelQueuedPromptQuery(CE_RequestId request_id);
std::string CE_RunQueuedRequestJsonString(CE_RequestId request_id);
