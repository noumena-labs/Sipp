#pragma once

#include <string>

#include "ffi_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int CE_InitPlugin(const char* model_path, const CE_InitConfig* config);
void CE_ClosePlugin();
int CE_GetRuntimeObservability(CE_RuntimeObservabilityMetrics* out_metrics);
int CE_ResetRuntimeObservability();
int CE_RunRequestStep(CE_RequestId request_id);
int CE_GetCompletedRequestStatus(CE_RequestId request_id);
int CE_GetCompletedRequestOutputSize(CE_RequestId request_id);
int CE_CopyCompletedRequestOutput(CE_RequestId request_id, char* buffer, int32_t capacity);
int CE_GetCompletedRequestErrorSize(CE_RequestId request_id);
int CE_CopyCompletedRequestError(CE_RequestId request_id, char* buffer, int32_t capacity);
int CE_ConsumeCompletedRequest(CE_RequestId request_id);
CE_RequestId CE_EnqueuePromptQuery(
    const char* context_key,
    const char* prompt,
    int n_tokens_predict,
    CE_TokenCallback on_token);
int CE_CancelQueuedPromptQuery(CE_RequestId request_id);

#ifdef __cplusplus
}
#endif

const char* CE_GetBackendObservabilityJsonString();
std::string CE_RunQueuedRequestJsonString(CE_RequestId request_id);
