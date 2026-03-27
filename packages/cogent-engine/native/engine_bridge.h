#pragma once

#include "ffi_types.h"

extern "C" {

int CE_InitPlugin(const char* model_path);
void CE_ClosePlugin();

int CE_SubmitActions(const char* agent_id, const ActionFFI actions[], int size);
int CE_SubmitThings(const char* agent_id, const ThingFFI things[], int size);
int CE_SubmitLocation(const char* agent_id, const LocationFFI* location);
int CE_SubmitGoals(const char* agent_id, const GoalFFI goals[], int size);
int CE_SubmitAgentState(const char* agent_id, const AgentStateFFI* state);

void CE_ProcessPromptQueryAsync(const char* context_key, const char* prompt,
                                int n_tokens_predict, CE_StringCallback callback,
                                CE_StringCallback stream);

void CE_PlanRoutineAsync(const char* agent_id, int steps, CE_PlanCallback callback);
void CE_FreePlan(ExecutionPlanFFI* plan);

}
