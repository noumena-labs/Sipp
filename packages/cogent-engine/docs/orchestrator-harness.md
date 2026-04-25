# Orchestrator Harness

`@noumena-labs/cogent-engine/orchestrator` is a config-driven director runtime.

It does not own world state, ticks, reducers, sensing, movement, or rendering. The host app owns the simulation and calls the director runtime when it wants model judgment about the current scenario state.

The package is responsible for:

- parsing `director.json`
- resolving static or runtime-supplied choices
- rendering deterministic prompts from config plus app-supplied inputs
- compiling output grammars for constrained shapes
- executing the task through `CogentEngine`
- parsing and validating the returned text against the declared output shape

## Public API

```ts
import {
  DirectorRuntime,
  createDirectorFromConfigUrl,
  parseDirectorConfig,
  type DirectorConfig,
  type DirectorRunRequest,
  type DirectorRunResult,
  type DirectorTaskPrompt,
} from '@noumena-labs/cogent-engine/orchestrator';
```

## Mental Model

```text
host app / game
  owns world state, ticks, reducers, rendering
        |
        |- builds inputs and optional runtime choices for a named task
        v
DirectorRuntime.run("resolve_conflict", request)
        |
        |- render system prompt from director.json
        |- render user prompt from task config + request inputs
        |- compile grammar from task.output when needed
        |- run model through CogentEngine
        `- parse + validate output by task.output shape
        v
shape-driven result for the host app to apply
```

## `director.json`

Minimal shape:

```json
{
  "id": "courtyard-director",
  "scenario": {
    "name": "Courtyard Snack",
    "summary": "A small social courtyard scene."
  },
  "director": {
    "role": "High-level scenario director",
    "objective": "Assess the supplied state and return concise results.",
    "instructions": [
      "Only reason from the supplied inputs."
    ]
  },
  "inputs": {
    "conflict": {
      "kind": "data",
      "description": "Conflict detector output from the host app."
    },
    "scene_brief": {
      "kind": "text",
      "description": "Facts to narrate."
    }
  },
  "tasks": {
    "resolve_conflict": {
      "purpose": "Resolve one conflict.",
      "instructions": [
        "Pick one listed ruling."
      ],
      "inputs": ["conflict"],
      "output": {
        "shape": "select_one",
        "choices": "runtime"
      }
    },
    "narrate_scene": {
      "purpose": "Write one short narration beat from the supplied observations.",
      "instructions": [
        "Use supplied observations only."
      ],
      "inputs": ["scene_brief"],
      "output": {
        "shape": "text",
        "minLength": 1,
        "maxLength": 160
      }
    }
  }
}
```

## Prompt Contract

Every task uses the same user-prompt sections:

```text
Task:
Resolve one conflict.

Instructions:
- Pick one listed ruling.

Response:
Select exactly one choice id. Output only the id.
Available choices:
- drop
- hold

Inputs:

conflict:
{
  "kind": "forced_drop"
}
```

Only the `Response` section varies by `task.output.shape`. Task purpose, instructions, inputs, input ordering, and media handling are shared for every task shape.

Input descriptions are rendered in the system prompt input glossary, not repeated in the user prompt.

## Output Shapes

- `select_one`: constrained to exactly one choice id.
- `select_many`: constrained to zero or more choice ids according to `min` and `max`.
- `select_slots`: constrained to one `slot=choice` line per configured slot.
- `text`: unconstrained plain text parsed with optional `minLength` and `maxLength` validation.
- `text_with_directives`: plain text with optional bracketed directive ids, parsed against configured directives.

`minLength` is a validation contract. It is not rendered as model-facing prompt text.

## Runtime

```ts
class DirectorRuntime {
  constructor(engine: CharacterAgentEngine, config: DirectorConfig, options?)

  getConfig(): DirectorConfig
  getSystemPrompt(): string
  getTaskGrammar(taskName: string, request?: DirectorRunRequest): string | undefined
  getTaskPrompt(taskName: string, request?: DirectorRunRequest): DirectorTaskPrompt

  run(taskName: string, request?: DirectorRunRequest): Promise<DirectorRunResult>
}
```

`DirectorRunResult` contains:

- `status`: `ok`, `aborted`, `timed_out`, `failed`, or `invalid_response`
- `text`: parsed text for text-like tasks
- `selections`: parsed selections for choice/directive tasks
- `rawText`: raw model output
- `errorMessage`: parse, validation, or runtime error when present

The package intentionally does not impose fallback business logic. If a task fails, the host app decides what to do next.

## Host App Responsibilities

- own the game loop and tick rate
- own world state and reducers
- decide when to query the director
- supply task inputs and runtime choices
- decide fallback behavior if the task fails
- own character brains, rendering, and scene bindings

The `apps/simulation` example shows one way to layer a world simulation on top of this generic director runtime.
