# CogentLM Rust Runtime

Rust-native CogentLM runtime workspace.

See `../../docs/engine-runtime-architecture.md` for the unified
Rust/Python/Node/browser runtime contract and naming rules.

This workspace is intentionally separate from the existing browser/Emscripten
package under `packages/cogentlm`. The sys crate builds the vendored llama.cpp
tree natively and exposes the C ABI plus a small CogentLM shim for features that
still live behind C++ APIs.

## Crates

- `cogentlm-sys`: CMake build, bindgen bindings, and C-compatible shims for chat templates and mtmd.
- `cogentlm-core`: typed runtime config, llama.cpp common-param/common-sampler integration, the scheduler-backed `InferenceRuntime`, and the `CogentEngine` owner-thread API.
- `cogentlm-cli`: proof-of-concept executable.
- `cogentlm-python`: PyO3 package binding for the native engine API.
- `cogentlm-node`: NAPI package binding for the native engine API.
- `cogentlm-browser`: unified Emscripten staticlib crate compiling both the inference scheduler FFI shims and browser OPFS GGUF split ingest logic into the WASM target.

## Example

```powershell
cargo run -p cogentlm-cli -- ..\..\LFM2.5-350M-Q4_K_M.gguf "Hello" --max-tokens 32
```

To exercise the scheduler/runtime path:

```powershell
cargo run -p cogentlm-cli -- ..\..\LFM2.5-350M-Q4_K_M.gguf "Hello" --max-tokens 32
```

For instruction-style prompts, apply the model chat template:

```powershell
cargo run -p cogentlm-cli -- ..\..\LFM2.5-350M-Q4_K_M.gguf "Describe browser LLM inference." --max-tokens 32 --chat
```

## Native Driver

`cogentlm-core::CogentEngine` is the native engine API layer. It loads
`InferenceRuntime` on a dedicated native thread and communicates through a
command channel:

```rust
use cogentlm_core::{ChatMessage, ChatRequest, CogentEngine, NativeRuntimeConfig, QueryOptions};

let engine = CogentEngine::load("model.gguf", NativeRuntimeConfig::default())?;
let answer = engine.chat(ChatRequest::new(vec![
    ChatMessage::user("Describe browser LLM inference."),
]).options(QueryOptions::default()))?;
println!("{}", answer.text);
engine.close()?;
```

This is the native-thread implementation of the engine-driver abstraction. The
browser implementation uses the same Rust runtime through the Emscripten wasm
export shim and keeps TypeScript as the browser storage, Worker, and app
adapter layer.

The shared runtime contract starts in `cogentlm_core::engine::protocol` and is
re-exported as `EngineState`, `EngineStats`, `EngineEvent`, and
`RequestResult`. The owner-thread driver also exposes explicit
`query_response` / `chat_response` methods when raw runtime details are
needed. The first canonical state view is
available through:

```rust
let state = engine.state()?;
println!("{:?}", state.stats);
```

The native engine also emits the shared event shape:

```rust
let events = engine.subscribe_events();
let result = engine.chat(ChatRequest::new(vec![
    ChatMessage::user("Describe browser LLM inference."),
]).options(QueryOptions::default()))?;
for event in events.try_iter() {
    println!("{event:?}");
}
println!("{:?}", result.stats);
```

`state()` is synchronous in the current driver. Live mid-query state is
available through `EngineEvent::State`; Python and Node expose the same
contract through `state()`, `drain_events()` / `drainEvents()`, `query()` /
`query`, and `chat()` / `chat`.

Run the runtime smoke example with a local GGUF to print backend/device
observability, raw-query/chat outputs, and runtime metrics:

```powershell
cargo run -q -p cogentlm-core --example phase3_smoke -- ..\..\LFM2.5-350M-Q4_K_M.gguf "Describe browser LLM inference." --max-tokens 32
```

For vision models, use the multimodal smoke example with the text model, the
matching mmproj GGUF, and an image file:

```powershell
cargo run -q -p cogentlm-core --example multimodal_smoke -- <MiniCPM-model.gguf> <MiniCPM-mmproj.gguf> ..\..\test.png "Describe this image in details" --max-tokens 128
```

The example defaults to chat-template mode and inserts the mtmd media marker
inside the user message before the prompt. Add `--raw` to test a raw prompt or
`--no-marker-in-message` to test the runtime's automatic marker prefix path.

The default build is CPU-only. `backend_before_load` / `backend_after_load`
show the compiled backends, discovered devices, memory, and whether
llama.cpp reports GPU offload support. To force CPU execution:

```powershell
cargo run -q -p cogentlm-core --example phase3_smoke -- ..\..\LFM2.5-350M-Q4_K_M.gguf "Describe browser LLM inference." --max-tokens 32 --gpu-layers 0
```

To build a native GPU-capable runtime, enable the matching Cargo feature and
allow offload with `--gpu-layers -1`:

```powershell
cargo run -q -p cogentlm-core --features cuda --example phase3_smoke -- ..\..\LFM2.5-350M-Q4_K_M.gguf "Describe browser LLM inference." --max-tokens 32 --gpu-layers -1
cargo run -q -p cogentlm-core --features vulkan --example phase3_smoke -- ..\..\LFM2.5-350M-Q4_K_M.gguf "Describe browser LLM inference." --max-tokens 32 --gpu-layers -1
```

The CLI forwards the same backend features:

```powershell
cargo run -q -p cogentlm-cli --features vulkan -- ..\..\LFM2.5-350M-Q4_K_M.gguf "Describe browser LLM inference." --max-tokens 32 --chat --gpu-layers -1
```

CUDA/Vulkan availability depends on the local toolchain and driver stack. If
switching features does not change the backend JSON, rebuild the sys crate:

```powershell
cargo clean -p cogentlm-sys
```

On Windows, CUDA linking uses `CUDA_PATH` or `CUDA_HOME` to find the CUDA
Toolkit `lib\x64` directory. The CUDA DLL directory must also be available on
`PATH` at runtime.

## Python Bindings

`cogentlm-python` is the first native package binding crate. It exposes the
`CogentEngine` API to Python through PyO3:

```python
from cogentlm import (
    ChatMessage,
    CogentEngine,
    ContextRuntimeConfig,
    ModelPlacementConfig,
    NativeRuntimeConfig,
    ObservabilityRuntimeConfig,
    QueryOptions,
)

engine = CogentEngine(
    "model.gguf",
    NativeRuntimeConfig(
        placement=ModelPlacementConfig(gpu_layers="all"),
        context=ContextRuntimeConfig(n_ctx=2048),
        observability=ObservabilityRuntimeConfig(runtime_metrics=True),
    ),
)
try:
    result = engine.chat(
        [ChatMessage.user("Describe browser LLM inference.")],
        QueryOptions(max_tokens=32),
    )
    print(result["text"].strip())
finally:
    engine.close()
```

Streaming uses the same native scheduler path and calls back into Python while
the binding releases the GIL around the blocking inference call:

```python
pieces = []

def on_tokens(batch: dict) -> None:
    pieces.append(batch["text"])
    print(batch["text"], end="", flush=True)

result = engine.chat(
    [ChatMessage.user("Describe browser LLM inference.")],
    QueryOptions(max_tokens=32),
    on_tokens=on_tokens,
)
assert "".join(pieces) == result["text"]
```

Install the local package for development with `uv`:

```powershell
cd crates\cogentlm-python
uv venv --python 3.9 .venv
uv sync --group dev
uv run maturin develop --features cuda
uv run --no-sync python examples\phase4_python_smoke.py ..\..\..\..\LFM2.5-350M-Q4_K_M.gguf "Describe browser LLM inference." --max-tokens 32 --gpu-layers -1
```

Use `--features native` or omit `--features cuda` for CPU-only local package
builds. After `maturin develop`, use `uv run --no-sync ...` for Python
commands so `uv` does not resync over the feature-specific native extension.

The Python wheel workflow lives at
`.github/workflows/python-wheels.yml`. It builds CPU wheels for Linux and
Windows, a Metal wheel for macOS, and has an opt-in CUDA wheel job for a
self-hosted Linux CUDA runner.

## Node Bindings

`cogentlm-node` is the initial NAPI binding crate. It mirrors the lower-level
engine surface exposed by Python and runs blocking native calls through
`AsyncTask`, so JavaScript callers get Promises rather than main-thread
blocking calls. `query` and `chat` return `RequestResult`; optional
`onTokens` callbacks receive batched `TokenBatch` values through a NAPI
thread-safe function:

```js
const { CogentEngine } = require("./index.js");

const engine = await CogentEngine.load("model.gguf", {
  placement: { gpuLayers: "all" },
  observability: { runtimeMetrics: true },
});

try {
  const result = await engine.chat(
    [{ role: "user", content: "Describe browser LLM inference." }],
    { maxTokens: 32 },
    (batch) => process.stdout.write(batch.text),
  );
  console.log(`\n${result.text.trim()}`);
} finally {
  await engine.close();
}
```

Build the local Node binding from the crate directory:

```powershell
cd crates\cogentlm-node
bun install
bun run build:cuda
node .\examples\node_smoke.mjs ..\..\..\..\Qwen3.5-0.8B-Q4_0.gguf "Describe browser LLM inference." --gpu-layers -1
```

## Character Harness Status

The character harness is not implemented in Rust yet. For now,
`character-agent.ts`, `action-grammar.ts`, and `action-parser.ts` remain in the
TypeScript package under `packages/cogentlm/src/character` and are still the
source of truth. The Rust runtime work is intentionally focused on native
engine bindings first; the character harness should be ported after the Python
and NAPI engine surfaces settle.

## Browser Ingest

The browser package uses a wasm32 WebGPU build with Rust GGUF ingest/splitting linked by Emscripten. Large monolithic GGUF files are split into OPFS-backed shards instead of relying on memory64.

The browser GGUF ingest path is only for the browser package. Native Rust, Python, and Node should keep using normal files or explicit shard arrays through their native loaders. Both the GGUF ingest/splitting logic and the main browser inference engine compile together under `cogentlm-browser` to produce a single static library linked into the wasm32 WebGPU package:

```powershell
bun run build:package:browser
```

That path exports Rust-backed `CE_BrowserCacheLayout`, `CE_GgufPlanSplitCount`, `CE_GgufSplitFile`, and `CE_GgufSplitStream` along with the core `CE_RustBrowserEngine` scheduler functions.
Remote browser model loading uses this for large monolithic GGUF URLs: the
download lands in a temporary OPFS file, Rust plans and writes split shards
through worker sync-access callbacks, and the temporary monolithic file is
deleted before runtime load.

Large local browser `File` imports use the same Rust-backed OPFS split path.
The selected file is copied into a temporary OPFS source, Rust writes
llama.cpp-compatible shards into OPFS, the temporary source is deleted, and the
registry stores local source metadata so repeated selection can reuse the
shards.
