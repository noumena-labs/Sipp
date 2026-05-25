# Director Harness

`cogentlm-browser/director` is a config-driven director runtime.

It does not own world state, ticks, reducers, sensing, movement, or rendering. The host app owns the simulation and calls the director runtime when it wants model judgment about current scenario state.

## 5-Line Quickstart

If you already have a loaded `CogentEngine` named `engine`, this is enough to ask the director for a decision.

```ts
import { createDirectorFromConfigUrl } from 'cogentlm-browser/director';
const { director } = await createDirectorFromConfigUrl({ configUrl: '/directors/courtyard/director.json', engine });
const result = await director.run('resolve_referee_event', { choices: [{ id: 'wait', label: 'Wait' }, { id: 'move', label: 'Move' }] });
if (result.status === 'ok') console.log(result.selections[0]?.id);
else console.warn(result.errorMessage);
```

Swap the URL and task name for your own `director.json`. The harness builds the prompt, constrains the answer, parses the result, and gives your app a clean decision object.

## Public API

```ts
import {
  DirectorRuntime,
  createDirectorFromConfigUrl,
  parseDirectorConfig,
  type DirectorConfig,
  type DirectorRunRequest,
  type DirectorRunResult,
  type DirectorRuntimeEngine,
} from 'cogentlm-browser/director';
```

`CogentEngine` implements `DirectorRuntimeEngine`; app authors usually do not implement it manually.

## Mental Model

```text
host app / game
  owns world state, ticks, reducers, rendering
        |
        |- builds inputs and optional runtime choices for a named task
        v
DirectorRuntime.run('resolve_referee_event', request)
        |
        |- render system prompt from director.json
        |- render user prompt from task config + request inputs
        |- compile grammar from task.output when needed
        |- run model through CogentEngine
        `- parse + validate output by task.output shape
        v
shape-driven result for the host app to apply
```

## director.json

```jsonc
{
  "id": "courtyard-director",
  "scenario": {
    "name": "Courtyard Snack",
    "summary": "A small social courtyard scene."
  },
  "director": {
    "role": "High-level scenario director",
    "objective": "Assess the supplied state and return concise results.",
    "instructions": ["Only reason from supplied inputs."]
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
    "resolve_referee_event": {
      "purpose": "Resolve one conflict.",
      "instructions": ["Pick one listed ruling."],
      "inputs": ["conflict"],
      "output": {
        "shape": "select_one",
        "choices": "runtime"
      }
    },
    "narrate_scene": {
      "purpose": "Write one short narration beat from the supplied observations.",
      "inputs": ["scene_brief"],
      "output": { "shape": "text" }
    }
  }
}
```

Rules:

- director `id`, task names, input names, and slot names must match `[A-Za-z0-9_-]+`
- choice ids must match `[A-Za-z0-9_.:-]+`
- task inputs are required at runtime unless the task does not list them
- runtime input kind must match the configured input kind

## Runtime

```ts
class DirectorRuntime {
  constructor(engine: DirectorRuntimeEngine, config: DirectorConfig, options?)

  run(taskName: string, request?: DirectorRunRequest): Promise<DirectorRunResult>
}
```

```ts
const { director } = await createDirectorFromConfigUrl({
  configUrl: '/directors/courtyard/director.json',
  engine,
  runtimeOptions: { maxOutputTokens: 128 },
});

const result = await director.run('resolve_referee_event', {
  inputs: {
    conflict: { kind: 'data', value: conflictPayload },
  },
  choices: [
    { id: 'drop', label: 'carrier drops', payload: { outcome: 'drop' } },
    { id: 'hold', label: 'carrier holds', payload: { outcome: 'hold' } },
  ],
  timeoutMs: 10_000,
});
```

## Result Shape

```ts
interface DirectorRunResult<TPayload = unknown> {
  status: RunStatus;
  text: string;
  selections: readonly DirectorSelection<TPayload>[];
  rawText: string;
  errorMessage?: string;
}
```

- `text` is populated for `text` and `text_with_directives`
- `selections` is populated for `select_one`, `select_many`, `select_slots`, and directives
- selection tasks return `text: ''`
- text tasks return `selections: []`
- `invalid_request` means missing inputs, bad input kinds, missing runtime choices, invalid choice ids, or oversized grammars
- `invalid_response` means model output did not parse against the declared shape

## Output Shapes

- `select_one`: exactly one choice id
- `select_many`: zero or more choice ids unless `min` is set; `max` cannot exceed available choices
- `select_slots`: one `slot=choice` line per configured slot
- `text`: unconstrained plain text
- `text_with_directives`: plain text with optional bracketed directive ids

Brackets are reserved for directive cues in `text_with_directives`.

The host app owns fallback behavior when a task does not return `ok`.
