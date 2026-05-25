# CogentLM Rust Runtime

Rust workspace for the CogentLM engine, native bindings, CLI, and browser wasm exports.

The Rust engine owns the shared runtime contract: model capabilities, query/chat/embed requests, scheduling, state, events, generation results, embedding results, and native backend configuration.

## Crates

- `cogentlm-sys`: llama.cpp CMake build, bindgen bindings, and C/C++ shims
- `cogentlm-engine`: core runtime, engine driver, lifecycle service, protocol types
- `cogentlm-cli`: command-line smoke and development tool
- `cogentlm-py`: Python binding through PyO3
- `cogentlm-napi`: Node binding through NAPI
- `cogentlm-wasm`: Emscripten static library used by `@noumena-labs/cogentlm-browser`
- `cogentlm-shard`: GGUF shard and split helpers

## Rust API

```rust
use cogentlm_engine::{
    ChatMessage, ChatRequest, ChatRole, CogentEngine, NativeRuntimeConfig, QueryOptions,
    QueryRequest,
};

let engine = CogentEngine::load("model.gguf", NativeRuntimeConfig::default())?;

let result = engine.chat(
    ChatRequest::new(vec![ChatMessage::new(ChatRole::User, "Say hi.")])
        .options(QueryOptions::default()),
)?;

println!("{}", result.text);
```

Use `chat()` when the loaded model has a GGUF chat template. It renders messages with that template, then runs text generation.

Use `query()` for raw text generation. It sends your prompt directly to the runtime and is the right API for plain prompts, custom templates, and encoder-decoder models such as T5.

Use `embed()` for vector output. It returns `EmbeddingResult`, not `GenerationResult`, and it never streams tokens.

```rust
use cogentlm_engine::{EmbedOptions, EmbedRequest};

let embedding = engine.embed(EmbedRequest {
    input: "document text".into(),
    options: EmbedOptions {
        normalize: true,
        context_key: Some("search".into()),
    },
})?;

println!("{} values via {:?}", embedding.values.len(), embedding.pooling);
```

Model-class behavior:

- Decoder-only text models: `chat()` if a chat template exists, otherwise `query()`.
- Encoder-decoder models: `query()` for source-to-target generation; `chat()` only if the model has a usable chat template.
- Encoder-only models: `embed()` only.
- Decoder-only embedding models: `embed()` only when loaded with embedding context.

`embed()` defaults to L2-normalized vectors. Normalization is ignored for rank-pooling reranker outputs.

## CLI

```bash
cargo run -p cogentlm-cli -- path/to/model.gguf "Say hi." --chat --max-tokens 64
```

CPU-only is the default. Enable a native GPU backend with Cargo features:

```bash
cargo run -p cogentlm-cli --features cuda -- path/to/model.gguf "Say hi." --chat --gpu-layers -1
cargo run -p cogentlm-cli --features vulkan -- path/to/model.gguf "Say hi." --chat --gpu-layers -1
```

Use `--gpu-layers 0` to force CPU execution.

## State And Events

```rust
let state = engine.state()?;
println!("{:?}", state.model);

let events = engine.subscribe_events();
let result = engine.query(QueryRequest::new("Raw prompt"))?;

for event in events.try_iter() {
    println!("{event:?}");
}
```

`GenerationResult` is returned by `query()` and `chat()`. `EmbeddingResult` is returned by `embed()`. They are intentionally separate result types.

## Python Binding

```python
from cogentlm import CogentEngine, ChatMessage, NativeRuntimeConfig, QueryOptions

engine = CogentEngine("model.gguf", NativeRuntimeConfig())
result = engine.chat(
    [ChatMessage("user", "Say hi.")],
    QueryOptions(max_tokens=64),
)
print(result["text"])
```

Streaming:

```python
def on_tokens(batch):
    print(batch["text"], end="", flush=True)

result = engine.chat(
    [ChatMessage("user", "Write a haiku.")],
    QueryOptions(max_tokens=64),
    on_tokens=on_tokens,
)
```

Local development:

```bash
cd crates/cogentlm-py
uv venv --python 3.9 .venv
uv sync --group dev
uv run maturin develop
```

## Node Binding

```js
const { CogentEngine } = require('./index.js');

const engine = await CogentEngine.load('model.gguf', {
  placement: { gpu_layers: 'all' },
});

const result = await engine.chat(
  [{ role: 'user', content: 'Say hi.' }],
  { maxTokens: 64 },
  (batch) => process.stdout.write(batch.text),
);

console.log(result.text);
```

Build locally:

```bash
cd crates/cogentlm-napi
bun install
bun run build
```

## Browser Wasm

`cogentlm-wasm` compiles the Rust engine and browser GGUF ingest helpers into the Emscripten static library consumed by `packages/cogentlm-browser`.

```bash
bun run build:package:browser
```

The browser package keeps browser-only responsibilities in TypeScript: `fetch`, `File`, OPFS, workers, and asset URL resolution. Runtime behavior stays in Rust.

## Test

```bash
cargo test
```

From the repo root:

```bash
cargo test --manifest-path packages/cogentlm-rs/Cargo.toml
```
