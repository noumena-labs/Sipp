/////////////////////////////////////////////////////////////////////////////////////////////////
// 
// ffi_types.h
// 
// - Handles the marshalling and conversion from managed and native data
// 
/////////////////////////////////////////////////////////////////////////////////////////////////


#pragma once
#include <cstddef>
#include <cstdint>


// ---------- FFI types (POD-only, no std::string/vector) ----------

// Forward declarations
struct ExecutionPlanFFI;

// Callback function types for async operations
typedef void (*CE_StringCallback)(const char* key, const char* result);
typedef void (*CE_PlanCallback)(ExecutionPlanFFI* plan);

struct ActionFFI {
    const char* Id;          // required: null-terminated
    const char* Name;
    const char* Description;
};

struct GoalFFI {
    const char* Id;
    const char* Name;
    const char* Description;
    float       Priority;
};

struct ThingFFI {
    const char* Id;
    const char* Name;
    const char* Description;
    float       DistanceToAgent;

    // Array of actions attached to this thing
    const ActionFFI* Actions; // pointer to first element
    int32_t          ActionsCount;
};

struct AgentStateFFI {
    const char* Name;
    const char* Persona;
    const ActionFFI ActiveAction;
    const ThingFFI ActiveThing;
};

struct LocationFFI {
	const char* LocationCompressed; // e.g., "building:floor:room"
};


struct PhaseFFI {
    int32_t     Step;
    ActionFFI   Action;
    ThingFFI    Thing;
};

struct ExecutionPlanFFI {
    const char* AgentId;
    const PhaseFFI* Phases; // plan-wide actions
    int32_t         PhasesCount;
};

#if INTPTR_MAX == INT32_MAX
static_assert(sizeof(ActionFFI) == 12, "ActionFFI layout mismatch");
static_assert(sizeof(GoalFFI) == 16, "GoalFFI layout mismatch");
static_assert(sizeof(ThingFFI) == 24, "ThingFFI layout mismatch");
static_assert(sizeof(AgentStateFFI) == 44, "AgentStateFFI layout mismatch");
static_assert(sizeof(LocationFFI) == 4, "LocationFFI layout mismatch");
static_assert(sizeof(PhaseFFI) == 40, "PhaseFFI layout mismatch");
static_assert(sizeof(ExecutionPlanFFI) == 12, "ExecutionPlanFFI layout mismatch");

static_assert(offsetof(ActionFFI, Id) == 0, "ActionFFI.Id offset mismatch");
static_assert(offsetof(ActionFFI, Name) == 4, "ActionFFI.Name offset mismatch");
static_assert(offsetof(ActionFFI, Description) == 8, "ActionFFI.Description offset mismatch");

static_assert(offsetof(GoalFFI, Id) == 0, "GoalFFI.Id offset mismatch");
static_assert(offsetof(GoalFFI, Name) == 4, "GoalFFI.Name offset mismatch");
static_assert(offsetof(GoalFFI, Description) == 8, "GoalFFI.Description offset mismatch");
static_assert(offsetof(GoalFFI, Priority) == 12, "GoalFFI.Priority offset mismatch");

static_assert(offsetof(ThingFFI, Id) == 0, "ThingFFI.Id offset mismatch");
static_assert(offsetof(ThingFFI, Name) == 4, "ThingFFI.Name offset mismatch");
static_assert(offsetof(ThingFFI, Description) == 8, "ThingFFI.Description offset mismatch");
static_assert(offsetof(ThingFFI, DistanceToAgent) == 12, "ThingFFI.DistanceToAgent offset mismatch");
static_assert(offsetof(ThingFFI, Actions) == 16, "ThingFFI.Actions offset mismatch");
static_assert(offsetof(ThingFFI, ActionsCount) == 20, "ThingFFI.ActionsCount offset mismatch");

static_assert(offsetof(AgentStateFFI, Name) == 0, "AgentStateFFI.Name offset mismatch");
static_assert(offsetof(AgentStateFFI, Persona) == 4, "AgentStateFFI.Persona offset mismatch");
static_assert(offsetof(AgentStateFFI, ActiveAction) == 8, "AgentStateFFI.ActiveAction offset mismatch");
static_assert(offsetof(AgentStateFFI, ActiveThing) == 20, "AgentStateFFI.ActiveThing offset mismatch");

static_assert(offsetof(LocationFFI, LocationCompressed) == 0, "LocationFFI.LocationCompressed offset mismatch");

static_assert(offsetof(PhaseFFI, Step) == 0, "PhaseFFI.Step offset mismatch");
static_assert(offsetof(PhaseFFI, Action) == 4, "PhaseFFI.Action offset mismatch");
static_assert(offsetof(PhaseFFI, Thing) == 16, "PhaseFFI.Thing offset mismatch");

static_assert(offsetof(ExecutionPlanFFI, AgentId) == 0, "ExecutionPlanFFI.AgentId offset mismatch");
static_assert(offsetof(ExecutionPlanFFI, Phases) == 4, "ExecutionPlanFFI.Phases offset mismatch");
static_assert(offsetof(ExecutionPlanFFI, PhasesCount) == 8, "ExecutionPlanFFI.PhasesCount offset mismatch");
#endif
