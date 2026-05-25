# CogentLM

CogentLM is a local inference runtime for GGUF models across browser and native targets. The same Rust engine contract powers:

- `cogentlm-browser`: browser/WebGPU runtime with OPFS model storage
- `cogentlm-engine`: native Rust engine
- `cogentlm-py`: Python binding
- `cogentlm-napi`: Node/NAPI binding
- `cogentlm-wasm`: Rust engine exports linked into the browser package

The public API is intentionally small: load a model, call `chat()`, `query()`, or `embed()`, observe state/events, then close the engine.

## Workspace

- `packages/cogentlm-browser`: npm browser package, worker/main-thread runtime, character and director helpers
- `packages/cogentlm-rs`: Rust workspace for native, Python, Node, CLI, and wasm engine code
- `packages/third_party/llama.cpp`: pinned llama.cpp checkout
- `apps/avatar`: character harness example
- `apps/benchmark`: browser benchmark and model smoke tests
- `apps/examples`: browser API examples
- `apps/proactive-ui`: browser UI experiment
- `apps/simulation`: director/simulation example

## Setup

```bash
git clone --recurse-submodules <repo-url> CogentLM
cd CogentLM
bun install
```

If the repo was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Browser Quickstart

```ts
import { CogentEngine } from '@noumena-labs/cogentlm-browser';

const engine = await CogentEngine.create();

await engine.models.load('/models/model.gguf');

const reply = await engine.chat([
  { role: 'user', content: 'Explain local browser inference in one sentence.' },
]);

console.log(reply.text);
await engine.close();
```

Use `chat()` when the loaded model has a GGUF chat template. It renders messages with that template, then runs text generation.

Use `query()` for raw text generation. It sends your prompt directly to the runtime, so it is the right API for plain prompts, custom templates, and encoder-decoder models such as T5.

```ts
const raw = await engine.query('<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n');
```

Use `embed()` for vector output:

```ts
const embedding = await engine.embed('search text', { normalize: true });
console.log(embedding.values.length, embedding.pooling);
```

Model-class behavior:

- Decoder-only text models: `chat()` if a chat template exists, otherwise `query()`.
- Encoder-decoder models: `query()` for source-to-target generation; `chat()` only if the model has a usable chat template.
- Encoder-only models: `embed()` only.
- Decoder-only embedding models: `embed()` only when loaded with embedding context.

## Native Rust Quickstart

```rust
use cogentlm_engine::{
    ChatMessage, ChatRequest, ChatRole, CogentEngine, NativeRuntimeConfig, QueryOptions,
};

let engine = CogentEngine::load("model.gguf", NativeRuntimeConfig::default())?;
let result = engine.chat(
    ChatRequest::new(vec![ChatMessage::new(ChatRole::User, "Say hi.")])
        .options(QueryOptions::default()),
)?;

println!("{}", result.text);
```

Run the CLI:

```bash
cargo run -p cogentlm-cli -- path/to/model.gguf "Say hi." --chat --max-tokens 64
```

## Browser Package

Common browser calls:

```ts
const state = engine.state();
const unsubscribe = engine.subscribeEvents((event) => console.log(event.type));

const streamed = await engine.chat([{ role: 'user', content: 'Write a haiku.' }], {
  maxTokens: 64,
  onTokens: (batch) => console.log(batch.text),
});

unsubscribe();
```

Model loading accepts URLs, `File` objects, GGUF shard arrays, and explicit model/projector pairs:

```ts
await engine.models.load({
  model: '/models/vision.gguf',
  projector: '/models/mmproj.gguf',
});
```

The browser package stores installed models in OPFS. Large monolithic GGUF files are split into OPFS shards before load.

## Character And Director Helpers

`@noumena-labs/cogentlm-browser/character` provides a small character runtime over `CogentEngine`.

```ts
import { createCharacterFromConfigUrl } from '@noumena-labs/cogentlm-browser/character';

const { character } = await createCharacterFromConfigUrl({
  configUrl: '/characters/aria/character.json',
  engine,
});

for await (const event of character.chat('Say hello.')) {
  if (event.kind === 'prose') console.log(event.text);
}
```

`@noumena-labs/cogentlm-browser/director` provides shape-driven task decisions for apps and simulations.

```ts
import { createDirectorFromConfigUrl } from '@noumena-labs/cogentlm-browser/director';

const { director } = await createDirectorFromConfigUrl({
  configUrl: '/directors/default/director.json',
  engine,
});

const result = await director.run('choose_action', {
  inputs: { state: { kind: 'data', value: worldState } },
  choices: [{ id: 'wait' }, { id: 'move' }],
});
```

## Build

```bash
bun run build:package
```

Build all browser wasm variants and validate the package:

```bash
bun run build:package:release
```

## Test

```bash
bun test packages/cogentlm-browser/src
cargo test --manifest-path packages/cogentlm-rs/Cargo.toml
```

## Architecture

Rust owns model classification, capabilities, scheduling, generation, embeddings, state, events, and native bindings. The browser TypeScript package owns browser objects: `fetch`, `File`, OPFS, workers, asset URLs, and user-facing browser helpers.

This keeps the cross-platform engine behavior in Rust while keeping browser integration thin and explicit.
