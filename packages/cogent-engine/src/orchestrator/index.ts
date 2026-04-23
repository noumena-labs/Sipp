//////////////////////////////////////////////////////////////////////////////
//
// orchestrator/index.ts
//
// - Barrel export for the `cogent-engine/orchestrator` subpath.
//
//////////////////////////////////////////////////////////////////////////////

export type {
  AgentIntent,
  AgentIntentKind,
  AgentPerception,
  ContestedObjectConflict,
  DirectorDecision,
  DirectorResolution,
  PerceivedAgent,
  PerceivedObject,
  ScenarioAgentSeed,
  ScenarioObjectSeed,
  ScenarioSeed,
  SimulationActionName,
  SimulationAgentState,
  SimulationObjectState,
  Vec2,
  WorldBounds,
  WorldConflict,
  WorldSnapshot,
} from './simulation-types.js';

export {
  DEFAULT_SIMULATION_EMOTION,
  SIMULATION_ACTION_NAMES,
  SIMULATION_ACTION_NAME_SET,
  assertCharacterActionsMatchSimulation,
  isSimulationActionName,
} from './simulation-character-actions.js';

export type {
  AgentActionEvent,
  AgentIntentEvent,
  AgentQueryEndEvent,
  AgentQueryStartEvent,
  AgentStateChangeEvent,
  DirectorConflictEvent,
  DirectorDecisionEvent,
  SimulationEvent,
  SimulationEventKind,
  SimulationEventListener,
  TickEndEvent,
  TickStartEvent,
  WorldNoteEvent,
} from './simulation-bus.js';
export { SimulationBus } from './simulation-bus.js';

export type { AgentOutput } from './agent-grammar.js';
export {
  defaultAgentOutput,
  getAgentGrammar,
  parseAgentOutput,
} from './agent-grammar.js';
export { getDirectorGrammar, parseDirectorOutput } from './director-grammar.js';

export type {
  SimulationAgentOptions,
  SimulationAgentQueryResult,
} from './simulation-agent.js';
export { SimulationAgent } from './simulation-agent.js';

export type { DirectorQueryResult, WorldDirectorOptions } from './world-director.js';
export { WorldDirector } from './world-director.js';

export {
  INTERACTION_RADIUS,
  applyDirectorDecision,
  applyTickFirstPass,
  stepMovement,
  type MutableWorldState,
  type TickReducerResult,
} from './world-reducer.js';

export {
  buildPerception,
  clampToBounds,
  vec2Direction,
  vec2Distance,
  type SensingOptions,
} from './sensing.js';

export type {
  AttachedSimulationAgent,
  WorldOrchestratorOptions,
} from './world-orchestrator.js';
export { WorldOrchestrator } from './world-orchestrator.js';

export type { CreateSimulationAgentOptions } from './create-simulation-agent.js';
export { createSimulationAgentFromConfigUrl } from './create-simulation-agent.js';
