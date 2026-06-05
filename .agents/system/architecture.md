# CogentLM Architecture Guide

The CogentLM monorepo is organized around public surfaces, internal runtime code, language bindings, demos, developer tools, and onboarding examples.

## 1. Rust Native Core (`crates/`)
The native engine is broken down into modular crates.
- **`crates/sys`**: Unsafe FFI bindings to the underlying C/C++ libraries (e.g., llama.cpp). Rust bridge declarations live under `src/`; CXX and llama.cpp shim files live under `native/`.
- **`crates/core`**: Low-level foundational Rust types and abstractions.
- **`crates/engine`**: The primary inference engine logic, memory management, and model lifecycle.
- **`crates/shard`**: GGUF cache planning and split-file writing utilities.
- **`crates/remote`**: Client-side transport for the CogentLM Remote Gateway Protocol (`/v1/query`, `/v1/chat`, and `/v1/embed`). App-facing remote clients depend on this crate, not on provider adapters.
- **`crates/gateway`**: Server-side CogentLM Remote Gateway implementation. Owns bearer auth, alias routing, normalized gateway routes, CORS setup, gateway-owned backends such as mock and hosted-local CogentEngine, and the `serve --config gateway.toml` binary.
- **`crates/gateway-providers`**: Server-side external provider adapter code for gateway use. Provider keys, upstream URLs, provider headers, and routing policy belong behind a gateway boundary, not in `CogentClient` or distributed app packages.
- **`lib/rust`**: The public Rust facade crate. Rust application examples and consumers should depend on this crate instead of internal runtime crates.
- **`apps/cli`**: The command-line interface for running the engine directly.
- **`xtask`**: The central build orchestrator (replaces `make`/`cmake` shell scripts).

## 2. Language Bindings (`bindings/`)
These directories contain the bridge code between the Rust core and other languages.
- **`bindings/node`**: N-API based bindings for Node.js using `@napi-rs/cli`.
- **`bindings/python`**: PyO3 and Maturin based bindings for Python.
- **`bindings/wasm`**: Emscripten compilation target for browser WebAssembly/WebGPU. Rust owns the JS-facing `CE_*` ABI with `#[no_mangle] extern "C"` exports under `src/`; CMake links that Rust staticlib with llama.cpp/ggml/mtmd backend objects and a small Emscripten JS shim under `native/emscripten/`.

## 3. Distribution Packages
- **`lib/web`**: Publishes `@noumena-labs/cogentlm` and the public `cogentlm` browser package. Includes high-level features like:
  - `character/`: Parsing and rendering agent personas and actions.
  - `orchestrator/`: The Director runtime for executing multi-step tasks.
  - `models/`: File system and OPFS management for downloading and caching models.
- **`lib/node`**: Publishes `@noumena-labs/cogentlm-server` and the public `cogentlm-server` Node package. Runtime JS, router files, package tests, and staging scripts live here.
- **`lib/python`**: Publishes Python `cogentlm`. The Python package files and tests live here while the PyO3 Rust crate remains in `bindings/python`.

## 4. Demos, Tools, And Examples
- **`demos/`**: Browser demos such as `chat`, `avatar`, `proactive-ui`, and `simulation`.
- **`tools/playground`**: Browser playground and Playwright browser runtime smoke harness.
- **`examples/node`**, **`examples/python`**, **`examples/rust`**, and **`examples/web`**: Runnable onboarding examples for public package surfaces.
