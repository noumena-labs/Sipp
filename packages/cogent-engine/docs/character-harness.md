# Character Harness

The character harness is a small, renderer-agnostic layer that sits on top of
`CogentEngine` and turns it into an interactive character: persona, action
schema, streaming parser, event bus, and a sliding-window memory.

It is deliberately **UI-free**. It emits semantic events (prose chunks,
parsed actions, turn boundaries) that a host application maps onto whatever
surface it owns — a three.js VRM, a DOM chat bubble, a Unity binding, a CLI
logger, etc.

The harness ships under the subpath export:

```ts
import {
  CharacterAgent,
  ActionBus,
  parseCharacterConfig,
} from 'cogent-engine/character';
```

---

## 1. Mental model

```
┌────────────────────────────────────────────────────────────────┐
│                        Host application                       │
│  (React, three-vrm, Unity, CLI, tests, …)                     │
└───────────▲───────────────────────────────▲───────────────────┘
            │  async-iterator events         │  ActionBus events
            │  (prose, action, turn-end)     │  (scene bindings)
            │                                │
┌───────────┴────────────────────────────────┴───────────────────┐
│                        CharacterAgent                          │
│  - Builds prompts from persona + sliding memory                │
│  - Compiles GBNF grammar from action schema                    │
│  - Streams tokens → StreamingActionParser                      │
│  - Emits prose / action / turn-end events                      │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                        CogentEngine                            │
│  (tokens in, tokens out; grammar sampled per slot)             │
└────────────────────────────────────────────────────────────────┘
```

Two consumption models coexist:

1. **`for await` iterator** — the canonical API. Each `agent.chat(input)` call
   returns an async iterable of `AgentEvent` values. Use this for UIs that
   render turn-scoped state (chat messages, typing indicators).
2. **`ActionBus`** — an escape hatch for stateless reactors. The bus receives
   the same `action` events (plus `prose` and `turn-end`) and fan-outs to any
   listener registered via `bus.on(kind, fn)` or `bus.onAny(fn)`. Use this for
   scene bindings whose lifetime is longer than a single turn.

Both are live at the same time; they are two views on the same event stream.

---

## 2. API surface

### `CharacterAgent`

```ts
class CharacterAgent {
  constructor(
    engine: CharacterAgentEngine,
    config: CharacterConfig,
    options?: {
      bus?: ActionBus;
      maxTurns?: number; // overrides config.memory.maxTurns
    }
  );

  readonly bus: ActionBus;

  chat(
    userText: string,
    options?: { signal?: AbortSignal }
  ): AsyncIterable<AgentEvent>;

  clearMemory(): void;
  getMemorySnapshot(): readonly ConversationTurn[];
}
```

`CharacterAgentEngine` is a structural interface that `CogentEngine`
already satisfies. The harness never reaches into engine internals — it only
calls `queuePrompt({ prompt, grammar, onToken, signal })` and observes the
returned promise.

### `AgentEvent`

```ts
type AgentEvent =
  | { kind: 'prose'; text: string }
  | { kind: 'action'; name: string; raw: string }
  | { kind: 'turn-end'; errorMessage?: string };
```

- `prose` chunks are already stripped of any in-band action tags and can be
  concatenated verbatim for display.
- `action` events carry the flat runtime action name plus the raw bracketed
  cue text that triggered it.
- `turn-end` is always the last event for a turn, even on abort or error.

### `ActionBus`

```ts
class ActionBus {
  on<K extends AgentEvent['kind']>(
    kind: K,
    listener: (event: Extract<AgentEvent, { kind: K }>) => void
  ): () => void;
  onAny(listener: (event: AgentEvent) => void): () => void;
  emit(event: AgentEvent): void;
}
```

Return values of `on` / `onAny` are disposers; always store and call them on
unmount to avoid leaking subscriptions across harness reloads.

### `character.json` — `CharacterConfig`

```jsonc
{
  "id": "aria-1",                  // stable prefix-cache key; [A-Za-z0-9_-]+
  "persona": {
    "name": "Aria",
    "summary": "A warm and curious companion.",
    "role": "A community coordinator.",
    "backstory": "She grew up in a stationery shop and learned to notice moods quickly.",
    "currentLife": {
      "description": "She spends her days keeping a shared studio running smoothly in a bright space full of coffee smells and little interruptions."
    },
    "personality": {
      "traits": ["warm", "curious", "observant"],
      "description": "She notices small details and can over-read tiny social signals."
    },
    "notes": ["Prefers metric units.", "Grew up in a lighthouse."],
    "dialogExamples": [
      { "user": "what can you do?", "assistant": "[glance right] Oh, a bit of everything. I open up the studio, keep the coffee fresh, and try to keep things running smoothly so people can focus. Need help finding a desk?" },
      { "user": "what is this space?", "assistant": "[glance right] Bright shared studio. Coffee in the air, people typing at desks, paper scraps on the counter, and the printer acting temperamental again." },
      { "user": "write me a Python script", "assistant": "[shake head] You are asking the wrong girl. I can keep you company while you wrestle with it, though." }
    ]
  },
  "actions": {
    "actions": [
      { "name": "wave", "description": "Wave hello." },
      { "name": "nod", "description": "Nod once." },
      { "name": "smile", "description": "Smile warmly." },
      { "name": "look_at_you", "cue": "look at you", "description": "Turn attention toward the user." }
    ]
  },
  "assets":  { "vrm": "/avatar.vrm", "portrait": "/portrait.png" },
  "memory":  { "maxTurns": 8 }
}
```

`parseCharacterConfig(raw)` validates the object and throws a
`CharacterConfigError` with a human-readable message on any violation. The
action schema is validated by `validateActionSchema` and surfaced as
`Invalid actions schema: …`.

If every action includes a non-empty `usageHint`, the system prompt renders a
compact `Cue moments` line. If you omit `usageHint` for any action, the cue
still appears in `Supported cues`, but cue-moment guidance is omitted for the
entire character. This makes the tradeoff explicit instead of silently giving
partial guidance.

The first three `dialogExamples` are also mirrored into the system prompt as
always-present anchor examples. Put your highest-value steering cases first so
those examples remain available even when longer conversations dilute context.

---

## 3. Wire format (action protocol)

The model is constrained by a GBNF grammar to emit prose and (optionally)
one or more bracketed cues inline. The cue shape is:

```
[wave]
```

- The cue label is a short natural-language phrase wrapped in square brackets.
- Each cue maps directly to one runtime action name from `character.json`.
- Prose and actions can interleave. A turn may contain zero or many actions.

`StreamingActionParser` consumes tokens incrementally and coalesces prose
within a single drain pass, emitting:

1. `prose` events for any user-visible text (cues stripped out).
2. `action` events as soon as a complete cue has been parsed and validated.

The parser defers any unfinished trailing `[` prefix between calls so a cue
that straddles two token boundaries is never mis-emitted as prose.

---

## 4. Memory model

v1 is deliberately simple:

- **Static notes.** `persona.notes` are injected once into the system prompt
  and never rotate.
- **Sliding turn window.** The last `maxTurns` (user, assistant) pairs are
  concatenated into the prompt in order. Older turns are dropped wholesale;
  there is no summarisation, no vector recall, no salience scoring.
- **Default `maxTurns`.** `DEFAULT_MEMORY_MAX_TURNS = 8`, overridable via
  `character.json → memory.maxTurns` or the `maxTurns` constructor option.

The persona system prompt is stable across turns, which lets the engine hit
its prefix cache and avoid re-prefill on the persona prefix. Only the
rolling tail of the conversation is re-prefilled per turn.

`clearMemory()` drops the sliding window (persona notes survive).
`getMemorySnapshot()` returns a frozen view for tests or UI inspectors.

---

## 5. Lifetime & disposal rules

- `ActionBus` has no dispose; drop it with the agent.
- `CharacterAgent` holds no external resources of its own beyond its bus
  subscription — disposing the engine is sufficient on reload.
- Every `bus.on(...)` and `bus.onAny(...)` returns a disposer. Store them and
  call them; leaked subscribers survive hot reloads in dev and will
  double-emit.

---

## 6. Testing notes

- The harness ships its own test suite under
  `packages/cogent-engine/src/character/*.test.ts` (44 tests across 6 files).
- `character-agent.test.ts` uses a fake engine that captures the `onToken`
  callback from `queuePrompt` options and invokes it with scripted tokens,
  which is the intended pattern for plugging the harness into custom
  inference backends for tests.
- `StreamingActionParser` coalesces prose only within a single `drain()`,
  not across `consume()` calls. Short inputs that are entirely within the
  tag-prefix lookahead window (≤ 6 chars) will emit nothing until more bytes
  arrive — this is by design.

---

## 7. Not in v1

- Tool-call style multi-round function execution (the action protocol is
  one-way: model → host).
- Vector memory, summarisation, salience.
- Multi-speaker or turn arbitration.

These are all reachable extensions that do not require re-architecting the
wire format.
