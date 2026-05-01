# Character Harness

`cogentlm/character` turns a loaded `CogentEngine` into a character runtime driven by `character.json`.

The host app still owns model loading, engine lifecycle, render assets, and business logic.

## 5-Line Quickstart

If you already have a loaded `CogentEngine` named `engine`, this is enough to make a character talk.

```ts
import { createCharacterFromConfigUrl } from 'cogentlm/character';
const { character } = await createCharacterFromConfigUrl({ configUrl: '/characters/aria/character.json', engine });
for await (const event of character.chat('Say hi in character.')) {
  if (event.kind === 'prose') document.body.textContent += event.text;
}
```

Swap the URL for your own `character.json`. The harness handles the character prompt, memory window, streaming text, and action cues for you.

## Public API

```ts
import {
  CharacterEventBus,
  CharacterRuntime,
  createCharacterFromConfig,
  createCharacterFromConfigUrl,
  parseCharacterConfig,
  type CharacterChooseResult,
  type CharacterRuntimeEngine,
} from 'cogentlm/character';
```

Advanced grammar/parser helpers live under:

```ts
import {
  compileActionGrammar,
  StreamingActionParser,
} from 'cogentlm/character/advanced';
```

## Mental Model

```text
character.json -> parseCharacterConfig -> CharacterRuntime
                                           |
                                queuePrompt(raw prompt, grammar)
                                           |
                                           v
                                     CogentEngine
                                           |
                                  streamed tokens
                                           |
                              text/action/turn-end events
```

`CharacterRuntime` has two primary APIs:

- `chat()` for streaming in-character conversational turns
- `choose()` for stateless constrained one-of-N decisions

## CharacterRuntime

```ts
class CharacterRuntime {
  constructor(
    engine: CharacterRuntimeEngine,
    config: CharacterConfig,
    options?: {
      bus?: CharacterEventBus;
      maxOutputTokens?: number;
      contextKey?: string;
    }
  )

  readonly bus: CharacterEventBus

  chat(userMessage: string, options?: { signal?: AbortSignal }): AsyncIterable<ChatEvent>

  choose(
    userMessage: string,
    options: {
      choices: readonly string[];
      signal?: AbortSignal;
      timeoutMs?: number;
      maxOutputTokens?: number;
    }
  ): Promise<CharacterChooseResult>

  getConfig(): CharacterConfig
  clearMemory(): void
  getMemory(): readonly ChatTurn[]
  getGrammarSource(): string
  getSystemPrompt(): string
}
```

`CogentEngine` implements `CharacterRuntimeEngine`; app authors usually do not implement it manually.

Only one streaming `chat()` turn may be active per character runtime. Starting a new `chat()` aborts the previous in-flight turn before the new one begins. Breaking out of the async iterator aborts the active turn.

## Loaders

```ts
const { character, config } = await createCharacterFromConfigUrl({
  configUrl: '/characters/aria/character.json',
  engine,
  bus: new CharacterEventBus(),
});
```

Use `createCharacterFromConfig({ config, engine })` when you already have a parsed config object.

## character.json

Minimal example:

```jsonc
{
  "id": "aria",
  "persona": {
    "name": "Aria",
    "summary": "A warm, curious companion.",
    "role": "A community coordinator.",
    "notes": [
      "Speak in first person and remain fully in character."
    ]
  },
  "actions": [
    {
      "id": "wave",
      "description": "Wave hello.",
      "usageHint": "warm greeting or goodbye"
    },
    {
      "id": "look_at_you",
      "cue": "look at you",
      "description": "Turn attention toward the user."
    }
  ],
  "memory": {
    "maxTurns": 8
  }
}
```

Rules:

- `id` must match `[A-Za-z0-9_-]+`
- `actions` is an array and may be empty for choose-only characters
- action `id` must match `[A-Za-z_][A-Za-z0-9_]*`
- action `cue` must not contain brackets, newlines, or control characters
- render assets do not belong in `character.json`
- character-specific style rules belong in `persona.notes`

Use `anchorExamples` for durable steering in the system prompt. Use `dialogExamples` for few-shot conversational turns replayed each chat turn.

## Events

```ts
type ChatEvent =
  | { kind: 'turn-start'; userMessage: string }
  | { kind: 'prose'; text: string }
  | { kind: 'action'; id: string; raw: string }
  | { kind: 'turn-end'; finalText: string; status: RunStatus; errorMessage?: string }
```

- `prose` is display text with cues stripped out
- `action.id` is the runtime action id from `actions[].id`
- `turn-end.status` is `ok`, `aborted`, `timed_out`, `failed`, `invalid_request`, or `invalid_response`

## choose()

Use `choose()` when the host app wants a strict answer from a fixed option list while still reusing the character persona prompt.

```ts
const result = await character.choose('What should you do next?', {
  choices: ['wait', 'wander', 'approach:aria', 'pick_up:banana'],
  timeoutMs: 10_000,
});

if (result.status === 'ok') {
  console.log(result.selection);
}
```

```ts
interface CharacterChooseResult {
  selection: string | null;
  status: RunStatus;
  errorMessage?: string;
  rawText: string;
}
```

`choose()` is stateless: it does not read or write chat memory.

## Memory Model

- `persona.notes` are static prompt material
- `memory.maxTurns` controls a plain sliding window of prior user/assistant pairs
- `clearMemory()` resets only the sliding window

No summarization, vector memory, or tool loop is built in.
