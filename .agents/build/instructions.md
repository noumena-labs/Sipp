# CogentLM Build Instructions

The CogentLM repository uses a custom Rust-based build orchestrator pattern called `xtask`. Do NOT use standard `cargo build` for anything other than basic Rust checks; always use the `xtask` orchestrator to build the project, as it automatically manages C++ dependencies, downloads toolchains (Vulkan SDK, Emscripten, Ninja), and injects the correct environment variables.

## Build Commands

From the root of the repository, execute the following commands depending on the target:

### 1. Native Rust Core
Builds the core workspace crates (excluding WASM and Python/Node bindings).
```bash
cargo xtask build core
```

### 2. Node Bindings
Builds the N-API Node bindings. You can optionally specify a hardware backend.
```bash
cargo xtask build node
# Or with a specific backend:
cargo xtask build node --backend vulkan
cargo xtask build node --backend cuda
cargo xtask build node --backend metal
```

### 3. Python Bindings
Builds the PyO3 bindings using `uv` and `maturin`.
```bash
cargo xtask build python
# Or with a specific backend:
cargo xtask build python --backend vulkan
```

### 4. Browser WASM/WebGPU
Compiles the engine using Emscripten to target WebAssembly. The `cogentlm-wasm` Rust staticlib owns the browser `CE_*` exports; the Emscripten link step preserves those exports while still linking llama.cpp/ggml/mtmd backend objects and the small host JS shim. This automatically downloads and activates the Emscripten SDK.
```bash
cargo xtask build wasm
```

### 5. Build Everything
Builds the default target set: core, WASM, Python CPU, Node CPU, and CLI CPU.
```bash
cargo xtask build all
```

## Troubleshooting
If a build fails stating missing CMake variables or SDKs, it is usually because the environment injection failed. The `xtask` orchestrator automatically downloads hermetic dependencies into `.build/toolchain/` at the root of the repo (e.g., `.build/toolchain/vulkan`, `.build/toolchain/emsdk`, `.build/toolchain/ninja`).

## Run And Test Commands

Use the `run` group for long-lived demos and non-test diagnostics. Use the
`test` group for white-box tests, interface tests, smoke checks, and coverage:

```bash
cargo xtask run demos serve chat
cargo xtask run demos build avatar
cargo xtask run llama backend-ops --backend cpu
cargo xtask test list
cargo xtask test unit suite xtask
cargo xtask test unit suite rust-crates --package cogentlm-core
cargo xtask test unit suite node-package --backend cpu
cargo xtask test smoke group local-model --backend cpu --model <model.gguf>
cargo xtask test smoke suite example-browser
cargo xtask test verify --target whitebox
```

Run `cargo xtask test list --cases` to inspect cataloged suites and cheap case
discovery. See `docs/testing.md` for the human-facing summary.
