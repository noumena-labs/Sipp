# CogentLM Architecture Guide

The CogentLM monorepo is organized to clearly separate the core Rust inference engine from the language-specific bindings and the high-level frontend applications.

## 1. Rust Native Core (`crates/`)
The native engine is broken down into modular crates.
- **`crates/sys`**: Unsafe FFI bindings to the underlying C/C++ libraries (e.g., llama.cpp). Rust bridge declarations live under `src/`; CXX and llama.cpp shim files live under `native/`.
- **`crates/core`**: Low-level foundational Rust types and abstractions.
- **`crates/engine`**: The primary inference engine logic, memory management, and model lifecycle.
- **`crates/shard`**: GGUF cache planning and split-file writing utilities.
- **`crates/providers`**: Wrappers and compatibility layers for external APIs (like OpenAI and Anthropic) to emulate local engine behavior.
- **`crates/cli`**: The command-line interface for running the engine directly.
- **`crates/xtask`**: The central build orchestrator (replaces `make`/`cmake` shell scripts).

## 2. Language Bindings (`bindings/`)
These directories contain the bridge code between the Rust core and other languages.
- **`bindings/node`**: N-API based bindings for Node.js using `@napi-rs/cli`.
- **`bindings/python`**: PyO3 and Maturin based bindings for Python.
- **`bindings/wasm`**: Emscripten compilation target for browser WebAssembly/WebGPU. Rust owns the JS-facing `CE_*` ABI with `#[no_mangle] extern "C"` exports under `src/`; CMake links that Rust staticlib with llama.cpp/ggml/mtmd backend objects and a small Emscripten JS shim under `native/emscripten/`.

## 3. NPM Packages (`packages/npm/`)
High-level JavaScript/TypeScript orchestration.
- **`@noumena-labs/cogentlm`**: The main JS package for browser environments. Includes high-level features like:
  - `character/`: Parsing and rendering agent personas and actions.
  - `orchestrator/`: The Director runtime for executing multi-step tasks.
  - `models/`: File system and OPFS management for downloading and caching models.

## 4. Applications (`apps/`)
Front-end applications and examples utilizing the engine.
- Contains sub-projects like `avatar`, `benchmark`, `proactive-ui`, etc.
