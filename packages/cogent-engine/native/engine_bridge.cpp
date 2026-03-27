#include "engine_manager.h"
#include "engine_bridge.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using noumena::cogentengine::Action;
using noumena::cogentengine::AgentState;
using noumena::cogentengine::CogentEngineManager;
using noumena::cogentengine::ExecutionPlan;
using noumena::cogentengine::Goal;
using noumena::cogentengine::Location;
using noumena::cogentengine::Phase;
using noumena::cogentengine::Thing;

namespace {
// This file bridges the stable C ABI used by wasm glue/TS to the C++ manager.

constexpr int kDefaultGpuLayers = 99;
constexpr int kStatusError = -1;
constexpr int kMaxPromptTokens = 2048;

std::mutex g_engineMutex;
std::shared_ptr<CogentEngineManager> g_engineManager;

std::string cstr_or_empty(const char* value) { return value ? value : ""; }

bool has_text(const char* value) { return value != nullptr && value[0] != '\0'; }
bool is_valid_prediction_tokens(int count) {
  return count > 0 && count <= kMaxPromptTokens;
}

std::shared_ptr<CogentEngineManager> acquire_engine_manager() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  return g_engineManager;
}

bool ffi_action_has_content(const ActionFFI& action) {
  return has_text(action.Id) || has_text(action.Name) || has_text(action.Description);
}

bool ffi_thing_has_content(const ThingFFI& thing) {
  return has_text(thing.Id) || has_text(thing.Name) || has_text(thing.Description);
}

Action from_ffi_action(const ActionFFI& src) {
  Action dst;
  dst.Id = cstr_or_empty(src.Id);
  dst.Name = cstr_or_empty(src.Name);
  dst.Description = cstr_or_empty(src.Description);
  return dst;
}

Goal from_ffi_goal(const GoalFFI& src) {
  Goal dst;
  dst.Id = cstr_or_empty(src.Id);
  dst.Name = cstr_or_empty(src.Name);
  dst.Description = cstr_or_empty(src.Description);
  dst.Priority = src.Priority;
  return dst;
}

std::vector<Action> from_ffi_actions(const ActionFFI* src, int count) {
  std::vector<Action> out;
  if (!src || count <= 0) {
    return out;
  }

  out.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.push_back(from_ffi_action(src[i]));
  }
  return out;
}

Thing from_ffi_thing(const ThingFFI& src) {
  Thing dst;
  dst.Id = cstr_or_empty(src.Id);
  dst.Name = cstr_or_empty(src.Name);
  dst.Description = cstr_or_empty(src.Description);
  dst.DistanceToAgent = src.DistanceToAgent;
  dst.Actions = from_ffi_actions(src.Actions, src.ActionsCount);
  return dst;
}

std::vector<Thing> from_ffi_things(const ThingFFI* src, int count) {
  std::vector<Thing> out;
  if (!src || count <= 0) {
    return out;
  }

  out.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.push_back(from_ffi_thing(src[i]));
  }
  return out;
}

std::vector<Goal> from_ffi_goals(const GoalFFI* src, int count) {
  std::vector<Goal> out;
  if (!src || count <= 0) {
    return out;
  }

  out.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.push_back(from_ffi_goal(src[i]));
  }
  return out;
}

AgentState from_ffi_agent_state(const AgentStateFFI& src) {
  AgentState dst;
  dst.Name = cstr_or_empty(src.Name);
  dst.Persona = cstr_or_empty(src.Persona);
  dst.hasActiveAction = ffi_action_has_content(src.ActiveAction);
  dst.hasActiveThing = ffi_thing_has_content(src.ActiveThing);

  if (dst.hasActiveAction) {
    dst.ActiveAction = from_ffi_action(src.ActiveAction);
  }
  if (dst.hasActiveThing) {
    dst.ActiveThing = from_ffi_thing(src.ActiveThing);
  }
  return dst;
}

Location from_ffi_location(const LocationFFI& src) {
  Location dst;
  dst.LocationCompressed = cstr_or_empty(src.LocationCompressed);
  return dst;
}

char* duplicate_c_string(const std::string& value) {
  const size_t bytes = value.size() + 1;
  char* out = static_cast<char*>(std::malloc(bytes));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, value.c_str(), bytes);
  return out;
}

void free_action_fields(ActionFFI& action) {
  std::free(const_cast<char*>(action.Id));
  std::free(const_cast<char*>(action.Name));
  std::free(const_cast<char*>(action.Description));
  action.Id = nullptr;
  action.Name = nullptr;
  action.Description = nullptr;
}

void free_thing_fields(ThingFFI& thing) {
  std::free(const_cast<char*>(thing.Id));
  std::free(const_cast<char*>(thing.Name));
  std::free(const_cast<char*>(thing.Description));

  if (thing.Actions) {
    ActionFFI* actions = const_cast<ActionFFI*>(thing.Actions);
    for (int i = 0; i < thing.ActionsCount; ++i) {
      free_action_fields(actions[i]);
    }
    std::free(actions);
  }

  thing.Id = nullptr;
  thing.Name = nullptr;
  thing.Description = nullptr;
  thing.Actions = nullptr;
  thing.ActionsCount = 0;
}

ActionFFI to_ffi_action(const Action& src) {
  ActionFFI dst{};
  dst.Id = duplicate_c_string(src.Id);
  dst.Name = duplicate_c_string(src.Name);
  dst.Description = duplicate_c_string(src.Description);
  return dst;
}

ThingFFI to_ffi_thing(const Thing& src) {
  ThingFFI dst{};
  dst.Id = duplicate_c_string(src.Id);
  dst.Name = duplicate_c_string(src.Name);
  dst.Description = duplicate_c_string(src.Description);
  dst.DistanceToAgent = src.DistanceToAgent;
  dst.ActionsCount = static_cast<int>(src.Actions.size());

  if (!src.Actions.empty()) {
    ActionFFI* actions = static_cast<ActionFFI*>(
        std::calloc(src.Actions.size(), sizeof(ActionFFI)));
    if (!actions) {
      dst.ActionsCount = 0;
      return dst;
    }

    for (size_t i = 0; i < src.Actions.size(); ++i) {
      actions[i] = to_ffi_action(src.Actions[i]);
    }
    dst.Actions = actions;
  }

  return dst;
}

PhaseFFI to_ffi_phase(const Phase& src) {
  PhaseFFI dst{};
  dst.Step = src.Step;
  dst.Action = to_ffi_action(src.Action);
  dst.Thing = to_ffi_thing(src.Thing);
  return dst;
}

ExecutionPlanFFI* to_ffi_plan(const ExecutionPlan& src) {
  // Ownership of the returned plan is transferred to the caller; free via CE_FreePlan.
  auto* plan = static_cast<ExecutionPlanFFI*>(std::calloc(1, sizeof(ExecutionPlanFFI)));
  if (!plan) {
    return nullptr;
  }

  plan->AgentId = duplicate_c_string(src.AgentId);
  plan->PhasesCount = static_cast<int>(src.Phases.size());

  if (src.Phases.empty()) {
    return plan;
  }

  auto* phases =
      static_cast<PhaseFFI*>(std::calloc(src.Phases.size(), sizeof(PhaseFFI)));
  if (!phases) {
    std::free(const_cast<char*>(plan->AgentId));
    std::free(plan);
    return nullptr;
  }

  for (size_t i = 0; i < src.Phases.size(); ++i) {
    phases[i] = to_ffi_phase(src.Phases[i]);
  }

  plan->Phases = phases;
  return plan;
}

}  // namespace

extern "C" {

int CE_InitPlugin(const char* model_path) {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  if (!has_text(model_path)) {
    return kStatusError;
  }

  if (g_engineManager) {
    return g_engineManager->IsReady() ? 0 : kStatusError;
  }

  auto manager =
      std::make_shared<CogentEngineManager>(cstr_or_empty(model_path), kDefaultGpuLayers);
  if (!manager || !manager->IsReady()) {
    return kStatusError;
  }

  g_engineManager = std::move(manager);
  return 0;
}

void CE_ClosePlugin() {
  std::lock_guard<std::mutex> lock(g_engineMutex);
  g_engineManager.reset();
}

int CE_SubmitActions(const char* agent_id, const ActionFFI actions[], int size) {
  auto manager = acquire_engine_manager();
  if (!manager || !agent_id || size < 0 || (size > 0 && actions == nullptr)) {
    return kStatusError;
  }
  return manager->SubmitAgentActions(cstr_or_empty(agent_id),
                                     from_ffi_actions(actions, size));
}

int CE_SubmitThings(const char* agent_id, const ThingFFI things[], int size) {
  auto manager = acquire_engine_manager();
  if (!manager || !agent_id || size < 0 || (size > 0 && things == nullptr)) {
    return kStatusError;
  }
  return manager->SubmitPerceivedThings(cstr_or_empty(agent_id),
                                        from_ffi_things(things, size));
}

int CE_SubmitLocation(const char* agent_id, const LocationFFI* location) {
  auto manager = acquire_engine_manager();
  if (!manager || !agent_id || !location) {
    return kStatusError;
  }
  return manager->SubmitAgentLocation(cstr_or_empty(agent_id),
                                      from_ffi_location(*location));
}

int CE_SubmitGoals(const char* agent_id, const GoalFFI goals[], int size) {
  auto manager = acquire_engine_manager();
  if (!manager || !agent_id || size < 0 || (size > 0 && goals == nullptr)) {
    return kStatusError;
  }
  return manager->SubmitAgentGoals(cstr_or_empty(agent_id),
                                   from_ffi_goals(goals, size));
}

int CE_SubmitAgentState(const char* agent_id, const AgentStateFFI* state) {
  auto manager = acquire_engine_manager();
  if (!manager || !agent_id || !state) {
    return kStatusError;
  }
  return manager->SubmitAgentState(cstr_or_empty(agent_id),
                                   from_ffi_agent_state(*state));
}

void CE_ProcessPromptQueryAsync(const char* context_key, const char* prompt,
                                int n_tokens_predict, CE_StringCallback callback,
                                CE_StringCallback stream) {
  auto manager = acquire_engine_manager();
  const char* callback_key = context_key ? context_key : "";
  if (!manager || !is_valid_prediction_tokens(n_tokens_predict)) {
    if (callback) {
      callback(callback_key, nullptr);
    }
    return;
  }

  const std::string result = manager->Prompt(
      cstr_or_empty(context_key), cstr_or_empty(prompt), n_tokens_predict,
      [callback_key, stream](const std::string& token) {
        if (stream) {
          stream(callback_key, token.c_str());
        }
      });

  if (callback) {
    callback(callback_key, result.c_str());
  }
}

void CE_PlanRoutineAsync(const char* agent_id, int steps, CE_PlanCallback callback) {
  if (!callback) {
    return;
  }

  auto manager = acquire_engine_manager();
  if (!manager || !agent_id || steps <= 0) {
    callback(nullptr);
    return;
  }

  const ExecutionPlan plan = manager->PlanAgentRoutine(cstr_or_empty(agent_id), steps);
  callback(to_ffi_plan(plan));
}

void CE_FreePlan(ExecutionPlanFFI* plan) {
  if (!plan) {
    return;
  }

  std::free(const_cast<char*>(plan->AgentId));
  if (plan->Phases) {
    PhaseFFI* phases = const_cast<PhaseFFI*>(plan->Phases);
    for (int i = 0; i < plan->PhasesCount; ++i) {
      free_action_fields(phases[i].Action);
      free_thing_fields(phases[i].Thing);
    }
    std::free(phases);
  }
  std::free(plan);
}

}  // extern "C"
