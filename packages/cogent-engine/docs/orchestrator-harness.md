# Orchestrator Harness

`@noumena-labs/cogent-engine/orchestrator` is a config-driven director runtime.

It does not own world state, ticks, reducers, sensing, movement, or
rendering. The host app owns the simulation and calls the director runtime
when it wants model judgment about the current scenario state.

The package is responsible for:

- parsing `director.json`
- rendering deterministic prompts from config plus app-supplied payload
- compiling the matching JSON response grammar
- executing the query through `CogentEngine`
- validating the returned JSON against the declared response schema

## Public API

```ts
import {
  DirectorRuntime,
  createDirectorFromConfigUrl,
  parseDirectorConfig,
  compileResponseGrammar,
  validateResponseValue,
  type DirectorConfig,
  type DirectorQueryPayload,
  type ResponseSchema,
} from '@noumena-labs/cogent-engine/orchestrator';
```

## Mental Model

```text
host app / game
  owns world state, ticks, reducers, rendering
        │
        ├─ builds payload for a named query
        ▼
DirectorRuntime.query("resolve_conflict", payload)
        │
        ├─ render system prompt from director.json
        ├─ render user message from query config + payload
        ├─ compile grammar from query.response
        ├─ run model through CogentEngine
        └─ parse + validate returned JSON
        ▼
structured result for the host app to apply
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
    "objective": "Assess the supplied state and return concise structured results.",
    "instructions": [
      "Only reason from the supplied payload."
    ]
  },
  "hooks": {
    "conflict": "Conflict detector output from the host app."
  },
  "queries": {
    "resolve_conflict": {
      "instructions": [
        "Pick one winner from the provided contenders or null to deny all."
      ],
      "hooks": ["conflict"],
      "response": {
        "type": "object",
        "properties": {
          "winnerAgentId": { "type": "string", "nullable": true, "maxLength": 64 },
          "note": { "type": "string", "maxLength": 160 }
        }
      }
    }
  }
}
```

## Runtime

```ts
class DirectorRuntime {
  constructor(engine: CharacterAgentEngine, config: DirectorConfig, options?)

  getConfig(): DirectorConfig
  getSystemPrompt(): string
  getGrammarSource(queryName: string): string

  query(
    queryName: string,
    payload: DirectorQueryPayload,
    options?: { signal?: AbortSignal }
  ): Promise<DirectorQueryResult>
}
```

`DirectorQueryResult` contains:

- `data`: validated JSON value or `null`
- `cancelled`: whether generation was cancelled
- `errorMessage`: parse/validation/runtime error when present
- `rawText`: raw model output

The package intentionally does not impose fallback business logic. If the
query fails, the host app decides what to do next.

## Response Schema Subset

Supported schema nodes:

- `object`
- `array`
- `string`
- `number`
- `boolean`
- `null`
- `nullable` on every non-null node
- `enum` and `maxLength` on strings
- `integer` on numbers
- `maxItems` on arrays

This is a pragmatic JSON-contract subset, not full JSON Schema.

## Host App Responsibilities

- own the game loop and tick rate
- own world state and reducers
- decide when to query the director
- decide fallback behavior if the query fails
- own character brains, rendering, and scene bindings

The `apps/simulation` example shows one way to layer a world simulation on
top of this generic director runtime.
