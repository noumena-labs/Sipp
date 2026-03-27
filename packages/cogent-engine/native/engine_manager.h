/////////////////////////////////////////////////////////////////////////////////////////////////
// 
// engine_manager.h
// 
// - Handles the management of model and context creations. 
// - Handles all incoming prompts and queries from the game engine. 
// 
/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(__CYGWIN32__)
#define COGENTENGINE_INTERFACE_API __stdcall
#define COGENTENGINE_INTERFACE_EXPORT __declspec(dllexport)
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(_WIN64) || defined(WINAPI_FAMILY)
#define COGENTENGINE_INTERFACE_API __stdcall
#define COGENTENGINE_INTERFACE_EXPORT __declspec(dllexport)
#elif defined(__MACH__) || defined(__ANDROID__) || defined(__linux__) || defined(LUMIN)
#define COGENTENGINE_INTERFACE_API
#define COGENTENGINE_INTERFACE_EXPORT __attribute__ ((visibility ("default")))
#else
#define COGENTENGINE_INTERFACE_API
#define COGENTENGINE_INTERFACE_EXPORT
#endif

#include <cstdint>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "llama.h"

// Forward declaration
struct llama_model;
struct llama_sampler;
struct llama_context;

namespace noumena::cogentengine
{


	struct COGENTENGINE_INTERFACE_EXPORT Observation
	{
		std::string Id;
		std::string Context;
	};

	struct COGENTENGINE_INTERFACE_EXPORT Determination
	{
		std::string Context;
		std::vector<Observation> Observations;
	};


	// Data structs
	struct COGENTENGINE_INTERFACE_EXPORT Action
	{
		std::string Id;
		std::string Name;
		std::string Description;
	};

	struct COGENTENGINE_INTERFACE_EXPORT Goal
	{
		std::string Id;
		std::string Name;
		std::string Description;
		float Priority = 0.0f;
	};


	struct COGENTENGINE_INTERFACE_EXPORT Thing
	{
		std::string Id;
		std::string Name;
		std::string Description;

		float DistanceToAgent = 0.0f;
		std::vector<Action> Actions;
	};

	struct COGENTENGINE_INTERFACE_EXPORT Location
	{
		std::string LocationCompressed; // e.g., "building:floor:room"
	};

	struct COGENTENGINE_INTERFACE_EXPORT AgentState
	{
		std::string Name;
		std::string Persona;

		// Todo: personality matrix / etc
		
		Action ActiveAction;
		bool hasActiveAction = false;

		Thing ActiveThing;
		bool hasActiveThing = false;
	};


	// Return
	struct COGENTENGINE_INTERFACE_EXPORT Phase
	{
		int Step;
		Action Action;
		Thing Thing;
	};

	struct COGENTENGINE_INTERFACE_EXPORT ExecutionPlan
	{
		std::string AgentId;
		std::vector<Phase> Phases;
	};




	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// MANAGER
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Internal structure to store all agent-related data
	struct AgentData
	{
		std::vector<Action> available_actions;
		std::vector<Thing> perceived_things;
		std::vector<Goal> goals;
		Location location;
		AgentState state;
	};

	struct ContextState {
		struct llama_context* ctx = nullptr;
		std::vector<llama_token> current_kv_tokens; // CPU mirror of VRAM
		int n_past = 0; // Head position
	};


	class COGENTENGINE_INTERFACE_EXPORT CogentEngineManager
	{
	private:
		std::unordered_map<std::string, ContextState> context_states_;
		std::vector<std::string> context_usage_order_;
		static constexpr size_t kMaxCachedContexts = 8;

		llama_model* primary_model_ = nullptr;
		llama_sampler* sampler_ = nullptr;
		mutable std::recursive_mutex operation_mutex_;

		// Agent state storage indexed by agent_id
		std::unordered_map<std::string, AgentData> agent_data_;


	///////////////////////////////////////////////////////////////////////////
	/// METHODS
	///////////////////////////////////////////////////////////////////////////

	public:

		CogentEngineManager(std::string model_path = "", int gpu_layers_n = 99);
		~CogentEngineManager();
		bool IsReady() const;

		std::string Prompt(
			std::string context_key, 
			std::string prompt, 
			int n_tokens_predict = 64,
			std::function<void(std::string)> onTokenReceived = nullptr);

		std::pair<int, std::string> ConstrainedResolve(std::string model_context_key, Determination determination);

		

		// Engine Interface
		
		int SubmitAgentActions(std::string agent_id, const std::vector<Action>& actions);
		int SubmitPerceivedThings(std::string agent_id, const std::vector<Thing>& things);
		int SubmitAgentGoals(std::string agent_id, const std::vector<Goal>& goals);

		int SubmitAgentLocation(std::string agent_id, const Location& locations);

		int SubmitAgentState(std::string agent_id, const AgentState& state);

		ExecutionPlan PlanAgentRoutine(std::string agent_id, int steps);


	private:
		bool EnsureContextSpace(ContextState& state, int new_tokens_needed, int n_ctx);
		void TouchContextKey(const std::string& context_key);
		void ReleaseContextState(const std::string& context_key);
		void EnforceContextLimit(const std::string& active_context_key);
	};
}
