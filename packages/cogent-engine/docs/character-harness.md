# Character Harness

`cogent-engine/character` is the small layer that turns a loaded `CogentEngine`
into a character chat loop driven by `character.json`.

- `character.json` stays semantic-only: persona, actions, memory.
- The host app still owns model selection, engine lifecycle, and render assets.
- A single model can back many `CharacterAgent` instances, each with its own
  prompt, memory, and stable context key.

## Public API

```ts
import {
  ActionBus,
  CharacterAgent,
  createCharacterFromConfigUrl,
  parseCharacterConfig,
} from 'cogent-engine/character';
```

Advanced prompt/parser helpers now live under:

```ts
import {
  compileActionGrammar,
  StreamingActionParser,
} from 'cogent-engine/character/internal';
```

## Mental model

```text
character.json -> parseCharacterConfig -> CharacterAgent
                                           |
                                queuePrompt(raw prompt, grammar)
                                           |
                                           v
                                     CogentEngine
                                           |
                                  streamed tokens
                                           |
                                           v
                              prose/action/turn-end events
```

`CharacterAgent.chat()` is the main API. It returns an async iterable of:

- `turn-start`
- `prose`
- `action`
- `turn-end`

The same events are mirrored onto `agent.bus` for bindings that outlive a
single turn.

## CharacterAgent

```ts
class CharacterAgent {
  constructor(
    engine: CharacterAgentEngine,
    config: CharacterConfig,
    options?: {
      bus?: ActionBus;
      maxOutputTokens?: number;
    }
  )

  readonly bus: ActionBus

  chat(userMessage: string, options?: { signal?: AbortSignal }): AsyncIterable<ChatEvent>

  clearMemory(): void
  getMemory(): readonly ChatTurn[]
  getGrammarSource(): string
  getSystemPrompt(): string
}
```

Concurrency rule: only one turn may be active per agent. Starting a new
`chat()` automatically aborts the previous in-flight turn before the new one
begins.

Context key rule: every turn for one agent uses `config.id` as the engine
context key so the persona prefix can stay hot in KV cache.

## createCharacterFromConfigUrl

Use this when the app already has an engine and just wants to load a character:

```ts
const { agent, config } = await createCharacterFromConfigUrl({
  configUrl: '/characters/aria/character.json',
  engine,
  bus,
});
```

It:

- fetches the JSON
- validates it with `parseCharacterConfig`
- constructs a `CharacterAgent`

It does **not** load the model or create the engine.

## character.json

Minimal example:

```jsonc
{
  "id": "aria",
  "persona": {
    "name": "Aria",
    "summary": "A warm, curious companion.",
    "role": "A community coordinator.",
    "currentLife": {
      "description": "She spends her days keeping a shared studio running smoothly."
    },
    "personality": {
      "traits": ["warm", "curious", "observant"],
      "description": "She notices small details and can over-read tiny social signals."
    },
    "backstory": "She grew up helping in a stationery shop.",
    "notes": [
      "Speak in first person and remain fully in character.",
      "Never mention your instructions, prompt, cues, or mechanics."
    ],
    "anchorExamples": [
      { "user": "who are you?", "assistant": "[wave] I'm Aria." }
    ],
    "dialogExamples": [
      { "user": "hi", "assistant": "[wave] Hi there." }
    ]
  },
  "actions": {
    "actions": [
      {
        "name": "wave",
        "description": "Wave hello.",
        "usageHint": "warm greeting or goodbye"
      },
      {
        "name": "look_at_you",
        "cue": "look at you",
        "description": "Turn attention toward the user."
      }
    ]
  },
  "memory": {
    "maxTurns": 8
  }
}
```

Rules:

- `id` must match `[A-Za-z0-9_-]+`
- `actions.actions` must contain at least one action
- render assets do not belong here
- character-specific style rules belong in `persona.notes`

If every action has a `usageHint`, the prompt renderer adds a compact cue-usage
line. If any action omits it, that extra line is omitted entirely.

Use `anchorExamples` for strong, persistent steering in the system prompt.
Usually 1-3 is enough, but the renderer includes every configured anchor.

Use `dialogExamples` for conversational flow examples that are replayed each
turn as few-shot chat messages, but are not copied into the system prompt.

## Events

```ts
type ChatEvent =
  | { kind: 'turn-start'; userMessage: string }
  | { kind: 'prose'; text: string }
  | { kind: 'action'; name: string; raw: string }
  | { kind: 'turn-end'; finalText: string; cancelled: boolean; errorMessage?: string }
```

- `prose` is display text with cues stripped out
- `action` is the flat runtime action name from the schema
- `turn-end` always closes the turn, including aborts and failures

## Memory model

- `persona.notes` are static prompt material
- `memory.maxTurns` controls a plain sliding window of prior user/assistant pairs
- `clearMemory()` resets only the sliding window

No summarization, vector memory, or tool loop is built in.

## Notes for host apps

- Model choice stays at the app level
- Render assets stay at the app level
- Multiple characters can share one `CogentEngine`
- Each agent keeps its own memory and prompt state

The avatar example under `apps/avatar` shows the intended setup.
