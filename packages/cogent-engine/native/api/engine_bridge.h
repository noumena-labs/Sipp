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
int CE_RunSchedulerTick();
int CE_RunSchedulerBurst(int32_t max_ticks,
                         int32_t max_completed_responses,
                         int32_t max_emitted_tokens,
                         CE_SchedulerBurstResult* out_result);
int CE_RunSchedulerBurstWithDeadline(int32_t max_ticks,
                                     int32_t max_completed_responses,
                                     int32_t max_emitted_tokens,
                                     int32_t max_duration_us,
                                     CE_SchedulerBurstResult* out_result);
int CE_RunRequestStep(CE_RequestId request_id);
int CE_GetCompletedRequestStatus(CE_RequestId request_id);
int CE_DrainCompletedRequestIds(CE_RequestId* buffer, int32_t capacity);
int CE_DrainRuntimeEvents(CE_RuntimeEvent* event_buffer,
                          int32_t event_capacity,
                          char* text_buffer,
                          int32_t text_capacity,
                          CE_RuntimeEventDrainResult* out_result);
int CE_GetCompletedRequestOutputSize(CE_RequestId request_id);
int CE_CopyCompletedRequestOutput(CE_RequestId request_id, char* buffer, int32_t capacity);
int CE_GetCompletedRequestErrorSize(CE_RequestId request_id);
int CE_CopyCompletedRequestError(CE_RequestId request_id, char* buffer, int32_t capacity);
int CE_GetCompletedRequestRuntimeObservability(CE_RequestId request_id,
                                               CE_RuntimeObservabilityMetrics* out_metrics);
int CE_ConsumeCompletedRequest(CE_RequestId request_id);
CE_RequestId CE_EnqueuePromptQuery(
    const char* context_key,
    const char* prompt,
    int n_tokens_predict,
    CE_TokenCallback on_token,
    const char* grammar);
CE_RequestId CE_EnqueuePromptWithMediaQuery(
    const char* context_key,
    const char* prompt,
    int n_tokens_predict,
    int32_t n_images,
    const uint8_t* images_flat_buffer,
    const int32_t* image_sizes,
    CE_TokenCallback on_token,
    const char* grammar);
const char* CE_GetMediaMarkerString();
const char* CE_GetChatTemplateString();
const char* CE_ApplyChatTemplateString(const char* messages_json,
                                       int add_assistant);
int CE_CancelQueuedPromptQuery(CE_RequestId request_id);

#ifdef __cplusplus
}
#endif

const char* CE_GetBackendObservabilityJsonString();
