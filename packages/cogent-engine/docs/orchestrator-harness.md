# Orchestrator Harness

`cogent-engine/orchestrator` is the rendering-agnostic layer that drives a
small multi-agent world simulation on top of `cogent-engine/character`.

- The world state, tick loop, perception, reducer, and director all live in
  this package.
- The host app owns the model, the scene graph, and the scenario definition.
- One `CogentEngine` can back many `SimulationAgent` instances plus a
  `WorldDirector`, each with its own context key so KV caches stay hot.

## Public API

```ts
import {
  WorldOrchestrator,
  WorldDirector,
  SimulationAgent,
  SimulationBus,
  createSimulationAgentFromConfigUrl,
  SIMULATION_ACTION_NAMES,
  assertCharacterActionsMatchSimulation,
  type WorldSnapshot,
  type SimulationEvent,
  type ScenarioSeed,
} from 'cogent-engine/orchestrator';
```

## Mental model

```text
           ┌──────────────────────────────────────────────────┐
           │ WorldOrchestrator (1.5 Hz tick loop)             │
           │                                                  │
           │  tick-start                                      │
           │     │                                            │
           │     ├─> maybeQueryOneAgent ── SimulationAgent ──┐│
           │     │                          (LLM, stateless) ││
           │     │                                           ▼│
           │     │                       intent (persists)    │
           │     ├─> applyTickFirstPass (movement + pickups)  │
           │     │       │                                    │
           │     │       └─> conflicts ─┐                     │
           │     │                      ▼                     │
           │     ├─> WorldDirector.resolveConflicts (LLM)     │
           │     │   or .narrate on cadence                   │
           │     │                                            │
           │     └─> tick-end (immutable WorldSnapshot)       │
           └──────────────────────────────────────────────────┘
```

Key rules:

- Only the orchestrator owns a timer. Agents and the director are JIT-queried.
- Ticks never overlap. If LLM inference exceeds the tick interval, the next
  tick simply waits.
- Only one agent is queried per tick (round-robin over agents whose intent
  is currently `null`). Intents persist across ticks until the reducer
  consumes them or they become invalid.
- The LLM only *suggests*; the reducer is authoritative for movement,
  pickups, and drops.
- Invalid agent output falls back to `{ wait, confused }`.
- Invalid director output falls back to a deterministic tie-break (first
  contender wins).

## WorldOrchestrator

```ts
class WorldOrchestrator {
  constructor(director: WorldDirector | null, options?: WorldOrchestratorOptions)

  readonly bus: SimulationBus
  readonly id: string

  getSnapshot(): WorldSnapshot
  getTickHz(): number
  setTickHz(hz: number): void
  isRunning(): boolean

  start(): void
  pause(): void
  step(): Promise<void>         // run exactly one tick
  dispose(): Promise<void>

  addAgent(agent: SimulationAgent, seed: ScenarioAgentSeed): void
  removeAgent(agentId: string): void
  upsertObject(seed: ScenarioObjectSeed): void
  removeObject(objectId: string): void
}

interface WorldOrchestratorOptions {
  id?: string;                      // scopes director context key
  bounds?: WorldBounds;             // default half-extent 8 (16×16 world)
  tickHz?: number;                  // default 1.5; clamped to [0.25, 20]
  directorCadenceTicks?: number;    // default 10
  initialDirectorNote?: string | null;
  bus?: SimulationBus;
}
```

## SimulationAgent

Thin, stateless per-tick wrapper over `CogentEngine`. One call = one query
with the current `AgentPerception`; grammar-constrained JSON output is
parsed into an `AgentIntent`.

```ts
class SimulationAgent {
  constructor(engine: SimulationAgentEngine, options: SimulationAgentOptions)

  readonly id: string               // runtime id, distinct from archetype id

  query(
    perception: AgentPerception,
    options?: { signal?: AbortSignal }
  ): Promise<SimulationAgentQueryResult>
}
```

Context key rule: each agent uses `agent:<agentId>` as its engine context
key, so the persona prefix stays cached across ticks.

## WorldDirector

```ts
class WorldDirector {
  constructor(engine: WorldDirectorEngine, options?: WorldDirectorOptions)

  narrate(snapshot: WorldSnapshot, options?: { signal?: AbortSignal }):
    Promise<DirectorQueryResult>

  resolveConflicts(
    snapshot: WorldSnapshot,
    conflicts: readonly WorldConflict[],
    options?: { signal?: AbortSignal }
  ): Promise<DirectorQueryResult>
}
```

Context key: `director:<orchestratorId>`.

## Events

Consume via `orchestrator.bus.on(listener)`:

```ts
type SimulationEvent =
  | { kind: 'tick-start';       tick: number; timeSeconds: number }
  | { kind: 'agent-query-start'; tick: number; agentId: string }
  | { kind: 'agent-query-end';   tick: number; agentId: string;
      intent: AgentIntent; emotion: SimulationActionName | null;
      cancelled: boolean; errorMessage?: string }
  | { kind: 'agent-intent';      tick: number; agentId: string; intent: AgentIntent }
  | { kind: 'agent-state';       tick: number; agent: SimulationAgentState }
  | { kind: 'agent-action';      tick: number; agentId: string;
      emotion: SimulationActionName }
  | { kind: 'director-conflict'; tick: number; conflicts: readonly WorldConflict[] }
  | { kind: 'director-decision'; tick: number; decision: DirectorDecision }
  | { kind: 'world-note';        tick: number; note: string }
  | { kind: 'tick-end';          tick: number; snapshot: WorldSnapshot };
```

## Fixed action set (v1)

```
thinking · curious · happy · confused · alert · frustrated · sleepy · celebrate
```

All scenario `character.json` files must expose **exactly** these action
names. The helper `assertCharacterActionsMatchSimulation(config)` throws a
descriptive error if they don't.

## character.json for agents

Same schema as the character harness, but `actions.actions` is fixed to the
eight names above. The app renders glyph overlays; nothing in
`character.json` references render assets.

## Notes for host apps

- Model choice, engine lifecycle, and scene graph all stay at the app level.
- All agents can share one `CogentEngine` with the director.
- Scenario definitions are plain data (`ScenarioSeed`), assembled in code.
- The app should subscribe to `tick-end` and mirror `snapshot` into its
  scene graph — no other rendering hook is needed.

The `apps/simulation` example shows the intended setup.
