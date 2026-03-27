#include <emscripten/emscripten.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "engine_bridge.h"

#define CE_UNITY_GLUE_VERSION "1.1.0"

namespace {

constexpr int kStatusFailure = -1;
constexpr int kStatusInvalidArguments = -2;
constexpr int kMaxPromptTokens = 2048;

bool is_valid_prediction_tokens(int token_count) {
  return token_count > 0 && token_count <= kMaxPromptTokens;
}

}  // namespace

extern "C" {
void UnityRegisterRenderingPlugin(void* loadFunc, void* unloadFunc) {
  (void)loadFunc;
  (void)unloadFunc;
}
}

namespace {

struct PromptSyncState {
  std::mutex mutex;
  std::atomic<bool> isComplete{false};
  bool hasResult = false;
  std::string response;
};

struct PlanSyncState {
  std::mutex mutex;
  std::atomic<bool> isComplete{false};
  ExecutionPlanFFI* planPtr = nullptr;
};

std::atomic<bool> g_isEngineInitialized{false};
// Serializes entry points exposed to JS so engine state transitions are consistent.
std::mutex g_unityApiMutex;
PromptSyncState g_promptState;
PlanSyncState g_planState;

char* duplicate_heap_string(const std::string& value) {
  char* out = static_cast<char*>(std::malloc(value.size() + 1));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}

void reset_prompt_sync_state() {
  std::lock_guard<std::mutex> lock(g_promptState.mutex);
  g_promptState.isComplete.store(false, std::memory_order_release);
  g_promptState.hasResult = false;
  g_promptState.response.clear();
}

void reset_plan_sync_state() {
  std::lock_guard<std::mutex> lock(g_planState.mutex);
  g_planState.isComplete.store(false, std::memory_order_release);
  g_planState.planPtr = nullptr;
}

void on_prompt_complete(const char* key, const char* result) {
  (void)key;
  std::lock_guard<std::mutex> lock(g_promptState.mutex);
  g_promptState.hasResult = (result != nullptr);
  g_promptState.response = result ? result : "";
  g_promptState.isComplete.store(true, std::memory_order_release);
}

void on_prompt_stream(const char* key, const char* result) {
  (void)key;
  (void)result;
}

void on_plan_complete(ExecutionPlanFFI* plan) {
  std::lock_guard<std::mutex> lock(g_planState.mutex);
  g_planState.planPtr = plan;
  g_planState.isComplete.store(true, std::memory_order_release);
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
const char* CE_Unity_GetVersion() { return CE_UNITY_GLUE_VERSION; }

EMSCRIPTEN_KEEPALIVE
int CE_Unity_IsInitialized() { return g_isEngineInitialized.load() ? 1 : 0; }

EMSCRIPTEN_KEEPALIVE
int CE_Unity_Init(const char* model_path) {
  std::lock_guard<std::mutex> lock(g_unityApiMutex);

  if (g_isEngineInitialized.load()) {
    return kStatusFailure;
  }

  if (!model_path || std::strlen(model_path) == 0) {
    return kStatusInvalidArguments;
  }

  const int init_status = CE_InitPlugin(model_path);
  if (init_status != 0) {
    return init_status;
  }
  g_isEngineInitialized.store(true);
  return 0;
}

EMSCRIPTEN_KEEPALIVE
void CE_Unity_Close() {
  std::lock_guard<std::mutex> lock(g_unityApiMutex);

  if (!g_isEngineInitialized.load()) {
    return;
  }

  CE_ClosePlugin();
  g_isEngineInitialized.store(false);
  reset_prompt_sync_state();
  reset_plan_sync_state();
}

EMSCRIPTEN_KEEPALIVE
char* CE_Unity_Prompt(const char* context_key, const char* prompt, int n_tokens) {
  {
    std::lock_guard<std::mutex> lock(g_unityApiMutex);
    if (!g_isEngineInitialized.load()) {
      return nullptr;
    }
    if (!is_valid_prediction_tokens(n_tokens)) {
      return nullptr;
    }
  }

  reset_prompt_sync_state();
  CE_ProcessPromptQueryAsync(context_key, prompt, n_tokens, on_prompt_complete,
                             on_prompt_stream);

  // JSPI allows cooperative waits here without freezing the browser main thread.
  while (!g_promptState.isComplete.load(std::memory_order_acquire)) {
    emscripten_sleep(1);
  }

  std::lock_guard<std::mutex> resultLock(g_promptState.mutex);
  return g_promptState.hasResult ? duplicate_heap_string(g_promptState.response)
                                 : nullptr;
}

EMSCRIPTEN_KEEPALIVE
void CE_Unity_PromptAsync(const char* context_key, const char* prompt, int n_tokens,
                          CE_StringCallback callback,
                          CE_StringCallback streamCB) {
  {
    std::lock_guard<std::mutex> lock(g_unityApiMutex);
    if (!g_isEngineInitialized.load()) {
      if (callback) {
        callback(context_key, nullptr);
      }
      return;
    }
  }

  if (!is_valid_prediction_tokens(n_tokens)) {
    if (callback) {
      callback(context_key, nullptr);
    }
    return;
  }

  CE_ProcessPromptQueryAsync(context_key, prompt, n_tokens, callback, streamCB);
}

EMSCRIPTEN_KEEPALIVE
int CE_Unity_SubmitActions(const char* agent_id, const ActionFFI* actions,
                           int count) {
  std::lock_guard<std::mutex> lock(g_unityApiMutex);
  if (!g_isEngineInitialized.load() || count < 0) {
    return kStatusFailure;
  }
  return CE_SubmitActions(agent_id, actions, count);
}

EMSCRIPTEN_KEEPALIVE
int CE_Unity_SubmitThings(const char* agent_id, const ThingFFI* things,
                          int count) {
  std::lock_guard<std::mutex> lock(g_unityApiMutex);
  if (!g_isEngineInitialized.load() || count < 0) {
    return kStatusFailure;
  }
  return CE_SubmitThings(agent_id, things, count);
}

EMSCRIPTEN_KEEPALIVE
int CE_Unity_SubmitLocation(const char* agent_id, const LocationFFI* location) {
  std::lock_guard<std::mutex> lock(g_unityApiMutex);
  if (!g_isEngineInitialized.load()) {
    return kStatusFailure;
  }
  return CE_SubmitLocation(agent_id, location);
}

EMSCRIPTEN_KEEPALIVE
int CE_Unity_SubmitGoals(const char* agent_id, const GoalFFI* goals, int count) {
  std::lock_guard<std::mutex> lock(g_unityApiMutex);
  if (!g_isEngineInitialized.load() || count < 0) {
    return kStatusFailure;
  }
  return CE_SubmitGoals(agent_id, goals, count);
}

EMSCRIPTEN_KEEPALIVE
int CE_Unity_SubmitAgentState(const char* agent_id, const AgentStateFFI* state) {
  std::lock_guard<std::mutex> lock(g_unityApiMutex);
  if (!g_isEngineInitialized.load()) {
    return kStatusFailure;
  }
  return CE_SubmitAgentState(agent_id, state);
}

EMSCRIPTEN_KEEPALIVE
ExecutionPlanFFI* CE_Unity_PlanRoutine(const char* agent_id, int steps) {
  {
    std::lock_guard<std::mutex> lock(g_unityApiMutex);
    if (!g_isEngineInitialized.load()) {
      return nullptr;
    }
    if (steps <= 0) {
      return nullptr;
    }
  }

  reset_plan_sync_state();
  CE_PlanRoutineAsync(agent_id, steps, on_plan_complete);

  // JSPI allows cooperative waits here without freezing the browser main thread.
  while (!g_planState.isComplete.load(std::memory_order_acquire)) {
    emscripten_sleep(1);
  }

  std::lock_guard<std::mutex> resultLock(g_planState.mutex);
  return g_planState.planPtr;
}

EMSCRIPTEN_KEEPALIVE
void CE_Unity_PlanRoutineAsync(const char* agent_id, int steps,
                               CE_PlanCallback callback) {
  {
    std::lock_guard<std::mutex> lock(g_unityApiMutex);
    if (!g_isEngineInitialized.load()) {
      if (callback) {
        callback(nullptr);
      }
      return;
    }
  }

  if (steps <= 0) {
    if (callback) {
      callback(nullptr);
    }
    return;
  }

  CE_PlanRoutineAsync(agent_id, steps, callback);
}

EMSCRIPTEN_KEEPALIVE
void CE_Unity_FreePlan(ExecutionPlanFFI* plan) {
  if (plan) {
    CE_FreePlan(plan);
  }
}

EMSCRIPTEN_KEEPALIVE
void CE_Unity_FreeString(char* str) {
  if (str) {
    std::free(str);
  }
}

EMSCRIPTEN_KEEPALIVE
void* CE_Unity_HeapAlloc(uint32_t size) { return std::malloc(size); }

EMSCRIPTEN_KEEPALIVE
void CE_Unity_HeapFree(void* ptr) {
  if (ptr) {
    std::free(ptr);
  }
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_SizeOfActionFFI() { return sizeof(ActionFFI); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_SizeOfGoalFFI() { return sizeof(GoalFFI); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_SizeOfThingFFI() { return sizeof(ThingFFI); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_SizeOfAgentStateFFI() { return sizeof(AgentStateFFI); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_SizeOfLocationFFI() { return sizeof(LocationFFI); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_SizeOfPhaseFFI() { return sizeof(PhaseFFI); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_SizeOfExecutionPlanFFI() { return sizeof(ExecutionPlanFFI); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ActionFFI_Id() { return offsetof(ActionFFI, Id); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ActionFFI_Name() { return offsetof(ActionFFI, Name); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ActionFFI_Description() {
  return offsetof(ActionFFI, Description);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_GoalFFI_Id() { return offsetof(GoalFFI, Id); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_GoalFFI_Name() { return offsetof(GoalFFI, Name); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_GoalFFI_Description() {
  return offsetof(GoalFFI, Description);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_GoalFFI_Priority() { return offsetof(GoalFFI, Priority); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ThingFFI_Id() { return offsetof(ThingFFI, Id); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ThingFFI_Name() { return offsetof(ThingFFI, Name); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ThingFFI_Description() {
  return offsetof(ThingFFI, Description);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ThingFFI_DistanceToAgent() {
  return offsetof(ThingFFI, DistanceToAgent);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ThingFFI_Actions() { return offsetof(ThingFFI, Actions); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ThingFFI_ActionsCount() {
  return offsetof(ThingFFI, ActionsCount);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_AgentStateFFI_Name() {
  return offsetof(AgentStateFFI, Name);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_AgentStateFFI_Persona() {
  return offsetof(AgentStateFFI, Persona);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_AgentStateFFI_ActiveAction() {
  return offsetof(AgentStateFFI, ActiveAction);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_AgentStateFFI_ActiveThing() {
  return offsetof(AgentStateFFI, ActiveThing);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_LocationFFI_LocationCompressed() {
  return offsetof(LocationFFI, LocationCompressed);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_PhaseFFI_Step() { return offsetof(PhaseFFI, Step); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_PhaseFFI_Action() { return offsetof(PhaseFFI, Action); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_PhaseFFI_Thing() { return offsetof(PhaseFFI, Thing); }

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ExecutionPlanFFI_AgentId() {
  return offsetof(ExecutionPlanFFI, AgentId);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ExecutionPlanFFI_Phases() {
  return offsetof(ExecutionPlanFFI, Phases);
}

EMSCRIPTEN_KEEPALIVE
uint32_t CE_Unity_Offset_ExecutionPlanFFI_PhasesCount() {
  return offsetof(ExecutionPlanFFI, PhasesCount);
}

}  // extern "C"
