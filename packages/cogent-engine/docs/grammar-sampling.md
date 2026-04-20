# Grammar Sampling

CogentEngine supports GBNF grammar-constrained sampling end-to-end: the
caller passes a grammar string, it flows through the TS bridge, across the
wasm boundary, and is applied as a fresh per-slot sampler in the native
inference runtime. This doc explains the design choices and the invariants
you must not break when extending it.

---

## 1. Why grammar sampling at all

The character harness uses GBNF to force the model to emit a **strict action
protocol** interleaved with free prose:

```
<action name="wave" args='{"mood":"happy"}'/>
```

Free-form function-calling (`{"tool_calls":[…]}` JSON) was rejected for
three reasons:

1. **Streaming parse.** Action tags can be recognised and dispatched as soon
   as the closing `/>` is produced; JSON requires brace balancing that
   defeats incremental parsing.
2. **Prose co-emission.** The protocol lets the model narrate and act in the
   same turn (`"I wave hello! <action name=\"wave\" args='{}'/>"`). A JSON
   envelope would either swallow the prose or require a second pass.
3. **Tiny grammars.** The full action schema for a character compiles to
   well under 2 KiB of GBNF, far below our size cap.

The grammar is compiled from the `actions` section of `character.json` by
`compileActionGrammar`. It constrains action `name` to the declared set and
validates each arg's type (string / number / enum) inline — mis-shaped
actions cannot be produced in the first place, so the parser doesn't need a
second validation pass.

---

## 2. The per-slot stateful sampler invariant

The grammar sampler is **stateful**: it tracks the current position in the
grammar's pushdown automaton. Two invariants follow:

1. **Never share a grammar sampler across slots.** A sampler that is being
   advanced by slot A has no meaningful state for slot B's partial
   generation.
2. **Never clone a grammar sampler from a prototype.** `llama_sampler_clone`
   does not guarantee a reset of the grammar automaton back to the `root`
   production; cloned grammar state silently produces garbage.

Therefore, when `request.grammar` is non-empty, the native runtime builds a
**fresh sampler chain per slot** with `llama_sampler_init_grammar(..., "root")`
prepended, instead of cloning the shared chain. See
`native/runtime/inference_runtime.cpp` around line 504:

```cpp
if (!slot->request->grammar.empty()) {
  // Fresh chain; do NOT clone the shared sampler.
  slot->sampler = llama_sampler_chain_init(sparams);
  llama_sampler *grammar_sampler = llama_sampler_init_grammar(
      grammar_vocab, slot->request->grammar.c_str(), "root");
  llama_sampler_chain_add(slot->sampler, grammar_sampler);
  // …penalties, top-k, top-p, temperature, dist…
}
```

Plain (non-grammar) requests continue to clone the shared sampler, which is
stateless and benefits from the shared configuration.

The chain order matters: the grammar sampler runs **first** so downstream
samplers (top-k, top-p, temperature) only see tokens the grammar allows.

---

## 3. 64 KiB grammar size cap

The TS bridge enforces a hard ceiling:

```ts
// src/wasm/wasm-bridge.ts
export const MAX_GRAMMAR_BYTES = 64 * 1024;
```

Any `queuePrompt({ grammar })` call whose grammar exceeds this UTF-8 byte
length is rejected at the bridge boundary before crossing into wasm. The
cap exists because:

- The wasm heap has a fixed runtime event drain buffer of the same size
  (`RUNTIME_EVENT_DRAIN_TEXT_BUFFER_SIZE_BYTES`), and grammars are copied
  through similar-sized transfer buffers. 64 KiB is comfortably above any
  hand-written character grammar (~1–2 KiB) and any reasonable code-gen
  grammar (tens of KiB), while keeping worst-case copy cost bounded.
- Grammar compilation inside llama.cpp has super-linear cost in grammar
  size; multi-megabyte grammars can stall a single request for seconds and
  hurt throughput for neighbouring slots.

If you genuinely need a larger grammar, raise both limits together — they
are intentionally equal.

---

## 4. TS → wasm transport

The grammar argument is plumbed as a plain string through every layer:

```
CharacterAgent
    → engine.queuePrompt({ grammar })       // TS runtime
        → WasmBridge.generate(..., grammar)  // validates size, passes string
            → wasm_exports / engine_bridge   // std::string grammar
                → InferenceRuntime::submit(..., std::string grammar)
                    → SlotState->request->grammar
```

Key points:

- The grammar string is **moved** into the `Request` struct on submission
  (`request.grammar = std::move(grammar)`), so there is exactly one owner.
- The sampler reads the C string pointer via `.c_str()` at slot activation;
  the string must outlive the slot, which it does because the slot holds a
  shared pointer to the request.
- Empty string means "no grammar": the runtime branches on
  `slot->request->grammar.empty()` to decide between the fresh-chain and
  clone paths.

---

## 5. Authoring grammars for action schemas

`compileActionGrammar(schema)` emits GBNF with this rough shape:

```
root       ::= prose (action prose)*
prose      ::= [^<]*
action     ::= "<action name=\"" action-name "\" args='" action-args "'/>"
action-name ::= "wave" | "nod" | "set_mood" | …
action-args ::= action-args-wave | action-args-nod | action-args-set_mood
action-args-wave     ::= "{}"
action-args-set_mood ::= "{\"mood\":\"" mood-value "\"}"
mood-value           ::= "happy" | "sad" | "surprised" | "angry" | "neutral"
…
```

Two things to know if you extend it:

- Each action produces its own `action-args-<name>` LHS. If you refactor,
  remember that downstream test string replacements must use `replaceAll`:
  the LHS name and an RHS reference both appear.
- Enum args are emitted as alternation of literal strings, so the model
  cannot invent new values. Prefer enums over free-form strings whenever
  the domain is closed.

---

## 6. Testing

- `action-grammar.test.ts` pins the compiled GBNF output against golden
  snippets; updating the compiler requires updating these.
- `wasm-bridge.test.ts` exercises the size cap (valid grammar passes;
  65 KiB grammar throws).
- End-to-end grammar-constrained generation is validated indirectly via
  `character-agent.test.ts` with a fake engine that asserts the `grammar`
  option was set when and only when the agent is in action mode.

Native verification requires a wasm build (`bun run build:wasm`); the TS
test suite can't exercise the llama.cpp sampler path directly.

---

## 7. Pitfalls to avoid

1. **Sharing a grammar sampler across slots.** Will desync state and emit
   tokens the grammar forbids.
2. **Cloning a grammar sampler.** Same failure mode; clone explicitly
   resets nothing about the automaton position.
3. **Passing an empty grammar string to mean "any grammar".** Empty means
   "no grammar"; passing `" "` (whitespace) will fail compilation inside
   llama.cpp.
4. **Growing the grammar beyond 64 KiB without growing the transfer
   buffer.** The bridge will reject it cleanly — good — but if you bump one
   without the other you'll see truncation/garbage on the native side.
5. **Reordering the sampler chain.** Grammar **must** run first. Running it
   after top-k/top-p can let the downstream samplers pick a disallowed
   token that the grammar would have rejected.
6. **Using `_` in rule names.** llama.cpp's GBNF parser accepts only
   `[a-zA-Z0-9-]` in rule identifiers, but our action-schema validator
   intentionally permits `_` in user-supplied action/arg names (LLMs have
   strong `snake_case` priors for function-calling). `compileActionGrammar`
   bridges the gap by sanitising `_` → `-` **only** in rule-name fragments;
   on-wire string literals like `name="set_mood"` and JSON arg keys like
   `"duration_ms"` remain verbatim so the action-parser can round-trip them.
   If two distinct identifiers collide after sanitisation (e.g. `set_mood`
   and `set-mood`), the compiler throws `ActionSchemaError` at compile
   time rather than producing an ambiguous grammar.

---

## 8. Identifier character sets — wire vs rule name

Two different charsets coexist in the action-schema plumbing:

| Surface | Charset | Enforced by |
|---|---|---|
| `ActionSpec.name`, `ActionArgSpec.name` (authoring / wire) | `[A-Za-z_][A-Za-z0-9_]*` | `IDENTIFIER_RE` in `action-schema.ts` |
| GBNF rule-name fragments (internal) | `[A-Za-z][A-Za-z0-9-]*` | llama.cpp GBNF parser |

The compiler keeps these two surfaces aligned by applying
`sanitizeRuleIdent` (a simple `_ → -` substitution) **only** at the four
rule-name construction sites in `action-grammar.ts`. Do not apply it to
`rawStringLiteral(action.name)` (wire tag name) or `jsonStringLiteral(arg.name)`
(JSON key) — those literals are consumed by `action-parser.ts`, which
expects them byte-for-byte as the schema declared them.
