# Native Interfaces Architecture

This guide explains how CogentLM crosses the Rust, C++, C, WebAssembly, Node.js,
and Python boundaries. It is intentionally high-level: use it to understand
where code belongs and how data moves before changing the lower level bridge
files.

## Mental Model

CogentLM keeps model execution in one Rust engine and uses narrow boundary
layers for each host environment.

```text
Node / Python / CLI
        |
        v
crates/engine
        |
        v
crates/engine/src/native_bridge.rs
        |
        v
crates/sys/src/bridge.rs            (cxx declarations)
        |
        v
crates/sys/native/cxx_bridge        (C++ RAII facade)
        |
        v
crates/sys/native/llama_shim        (C ABI shim)
        |
        v
third_party/llama.cpp, ggml, mtmd
```

Browser builds add one more host-facing layer above the same engine:

```text
TypeScript package / Emscripten Module
        |
        v
bindings/wasm/src/exports.rs                      (CE_* exports)
        |
        v
bindings/wasm/src/engine/mod.rs                   (BrowserEngine)
        |
        v
crates/engine -> crates/sys -> llama.cpp
```

## Native Core Boundary

`crates/sys` owns the low-level bridge to llama.cpp. Higher-level crates should
not reach into llama.cpp directly.

The important files are:

- `crates/sys/src/bridge.rs`: the `#[cxx::bridge]` declaration. This is the
  single Rust declaration of the C++ interface exposed to the engine.
- `crates/sys/native/cxx_bridge/cogent_cxx.h`: C++ declarations that match the
  CXX bridge.
- `crates/sys/native/cxx_bridge/cogent_cxx.cpp`: C++ implementation of the
  bridge facade.
- `crates/sys/native/llama_shim/cogent_shim.h`: a C ABI wrapper around selected
  llama.cpp/common/mtmd behavior.
- `crates/sys/native/llama_shim/cogent_shim.cpp`: the shim implementation that
  calls llama.cpp C and C++ APIs.
- `crates/engine/src/native_bridge.rs`: the safe, crate-private engine facade
  over `cogentlm_sys::bridge`.

The repo does not currently use Rust `bindgen` to generate raw bindings from
`llama.h`. Instead, it uses a hand-curated CXX bridge and a small set of Rust
type aliases such as `llama_token` and `llama_seq_id`. This keeps the Rust API
small and avoids exposing unstable llama.cpp internals through the rest of the
workspace.

### Why There Are Two Native C++ Layers

The CXX facade and the C shim solve different problems.

`cogent_cxx.*` is shaped for Rust. It uses CXX-compatible types such as
`rust::Str`, `rust::Vec`, `rust::String`, `std::unique_ptr`, and opaque C++
classes. It also converts native failures into `std::runtime_error`, which CXX
maps back to Rust `Result` for fallible bridge functions.

`cogent_shim.*` is shaped for llama.cpp. It isolates direct use of
`common_params`, `common_sampler`, chat templates, backend registration, mtmd,
and raw `llama_context` operations. It presents plain C functions, fixed-width
integers, opaque pointers, and explicit free functions.

This split lets Rust depend on a stable, reviewable API while still allowing the
shim to adapt to upstream llama.cpp churn.

## Core Native Types

The CXX bridge exposes three opaque native handles.

`NativeRuntime` owns a loaded llama model/context plus auxiliary state such as
chat templates and the optional mtmd multimodal context. It is responsible for
runtime metadata, tokenization, chat template rendering, decode/encode, KV
sequence state, embeddings, sampler attachment, backend synchronization, and
multimodal image evaluation.

`NativeBatch` owns and manages a `llama_batch`. Rust fills it token by token
through bridge methods, and the runtime passes it to `decode` or `encode`.
Capacity management stays in C++ because llama batch storage is native memory.

`CommonSampler` owns the llama.cpp common sampler stack. The engine can sample
through it, accept generated tokens, reset it, or attach/detach backend sampling
to a runtime sequence.

Rust engine code should use `NativeRuntimeHandle`, `NativeBatchHandle`, and
`SamplerHandle` from `crates/engine/src/native_bridge.rs`, not raw
`cogentlm_sys::bridge` types. That facade centralizes null checks, pinning,
error mapping, and test-only empty handles.

## Build Flow

Native builds are coordinated by `cargo xtask` and the `crates/sys` build
script.

1. `crates/sys/build.rs` delegates to `crates/sys/build_support`.
2. For native targets, `build_support/cmake.rs` builds llama.cpp, ggml, mtmd,
   and `cogent_shim` through CMake.
3. `build_support/cxx.rs` runs `cxx_build::bridge("src/bridge.rs")`, compiling
   generated CXX glue plus `native/cxx_bridge/cogent_cxx.cpp`.
4. `build_support/link.rs` links the CMake outputs and target-specific system
   libraries.
5. `crates/engine` links against `cogentlm-sys` and exposes safe runtime APIs.

For Emscripten targets, `crates/sys` only compiles the CXX bridge during Cargo's
Rust staticlib build. The final browser CMake step later links that Rust
staticlib with llama.cpp, mtmd, WebGPU support, `cogent_shim`, and the
Emscripten host shim.

Use the repository build commands rather than invoking CMake directly. The
normal entry points are:

- `cargo xtask build core` for native Rust.
- `cargo xtask build node` for N-API bindings.
- `cargo xtask build python` for PyO3/Maturin bindings.
- `cargo xtask build wasm` for browser WebAssembly/WebGPU artifacts.

## Browser/Wasm Boundary

`bindings/wasm` does not contain a custom C++ browser host bridge. The
TypeScript package calls stable `CE_*` symbols on the Emscripten module, and
those symbols are implemented directly in Rust as `#[no_mangle] extern "C"`
functions in `bindings/wasm/src/exports.rs`.

`exports.rs` owns browser ABI concerns that used to sit in C++: the current
engine singleton for `CE_Init`/request APIs, explicit smoke-test handles for
`CE_RustBrowserEngineCreate`/`Id`/`Close`, C string parsing, pointer and length
validation, byte and `f32` slice copying, owned string allocation, last-error
copying, backend observability JSON enrichment, and the default `LLAMA_CACHE`
setup.

`bindings/wasm/src/abi.rs` contains the C-compatible structs that TypeScript
reads directly from WASM memory. Keep their `#[repr(C)]` layout and size
assertions stable unless the TypeScript reader is updated at the same time.
`CE_RequestId` remains a `u32`; runtime metrics are 88 bytes; scheduler loop
results are 16 bytes.

`bindings/wasm/src/ingest/mod.rs` adapts streamed GGUF reads and shard writes
from raw Emscripten callback function pointers. JS obtains those pointers with
`Module.addFunction`, passes explicit `user_data`, and Rust calls the callbacks
through `unsafe extern "C"` function pointer types after the exported entrypoint
has validated the callback set.

`bindings/wasm/native/emscripten/ce_host.js` is the only custom browser-host
native shim. It provides `ce_native_yield`, which calls
`Module._ce_yield_drain()` so the Rust scheduler can synchronously drain token
bytes into the shared-memory streaming ring.

The browser build has two linked pieces:

1. Cargo builds `cogentlm-wasm` as a Rust staticlib containing the `CE_*`
   exports and Rust browser runtime code.
2. Emscripten/CMake links that staticlib with llama.cpp, ggml WebGPU, mtmd,
   `cogent_shim`, and `ce_host.js`, then preserves the `CE_*` symbols used by
   the TypeScript package.

## Node And Python Bindings

Node and Python bindings are high-level host bindings over `cogentlm-engine`.
They do not call llama.cpp or the CXX bridge directly.

`bindings/node/src/lib.rs` uses napi-rs. The `#[napi]` macros and
`napi::bindgen_prelude` generate the JavaScript-facing native module surface:
configuration objects, `CogentEngine`, `ModelService`, explicit gateway
descriptors, async tasks, event draining, and small backend helpers. Native model
execution still flows through `cogentlm-engine` and then `native_bridge.rs`.

`bindings/python/src/lib.rs` uses PyO3 and Maturin. The `#[pyclass]`,
`#[pymethods]`, `#[pyfunction]`, and `#[pymodule]` surfaces mirror the same
engine concepts for Python. Long-running work releases the GIL with
`allow_threads`, then maps engine, model, gateway endpoint, and provider errors
into Python exceptions.

In this repo, "bindgen" usually means these generated host-language binding
surfaces, especially napi-rs' bindgen prelude. It does not mean generated
llama.cpp C header bindings.

## Ownership And ABI Rules

Keep these rules in mind when changing boundary code:

- Rust-facing C++ objects cross CXX as opaque `std::unique_ptr` handles.
- CXX methods that mutate native objects take pinned mutable references on the
  Rust side.
- Any C string passed into llama.cpp is copied and checked for interior NUL
  bytes before becoming a `const char *`.
- Native strings returned through C ABIs are heap allocated and must have a
  matching free path. Browser strings returned to JS are released with
  `CE_FreeString`.
- FFI-facing integers use fixed-width types at native, C ABI, and WASM
  boundaries.
- Browser structs in `bindings/wasm/src/abi.rs` have compile-time size checks
  because TypeScript reads their memory layouts directly.
- Callbacks crossing the WASM boundary carry explicit `user_data`; Rust should
  not assume anything about its shape.
- Null pointers, invalid lengths, invalid counts, and missing buffers are
  rejected at the first boundary that can validate them.

## Adding A New Native Capability

Most native changes follow this path:

1. Decide whether the feature belongs in `crates/engine` or really needs a new
   llama.cpp bridge call. Prefer engine-level composition when possible.
2. If the feature needs llama.cpp/common/mtmd internals, add or adjust a focused
   function in `cogent_shim.h` and `cogent_shim.cpp`.
3. Add the Rust-shaped method in `cogent_cxx.h` and `cogent_cxx.cpp`.
4. Add the matching declaration to `crates/sys/src/bridge.rs`.
5. Add a safe wrapper in `crates/engine/src/native_bridge.rs`.
6. Use that wrapper from the runtime, scheduler, lifecycle, or model-service
   module that owns the behavior.
7. Expose the behavior through Node, Python, or Wasm only if it is part of those
   public APIs.
8. Update CMake/build rerun triggers if new native files are introduced.
9. Add the narrowest relevant tests around the Rust owner of the behavior, not
   around every bridge layer unless the bridge behavior itself is the risk.

For browser-only host APIs, start at `bindings/wasm/src/exports.rs`. Update
`bindings/wasm/src/abi.rs` if the memory layout changes, preserve
`CE_FreeString` ownership rules for Rust-owned strings, add the export root to
`bindings/wasm/CMakeLists.txt`, and update the TypeScript package that calls
the exported `CE_*` symbol.

For Node or Python-only APIs, start in the binding file and map to existing
engine APIs. Avoid duplicating engine behavior in the binding layer.

## Where To Debug

Use this file map to narrow investigation quickly:

- Load, backend selection, decode/encode failures: start in
  `crates/engine/src/runtime/inference_runtime`, then follow
  `native_bridge.rs` into `cogent_cxx.cpp`.
- Llama.cpp parameter parsing, sampler JSON, chat templates, mtmd, backend
  observability: inspect `cogent_shim.cpp`.
- Linker or backend build failures: inspect `crates/sys/build_support` and
  `crates/sys/CMakeLists.txt`.
- Browser `CE_*` export issues: inspect `bindings/wasm/src/exports.rs` and
  `bindings/wasm/CMakeLists.txt`.
- Browser Rust handle or string/buffer copy issues: inspect
  `bindings/wasm/src/exports.rs`.
- GGUF streamed ingestion in the browser: inspect
  `bindings/wasm/src/ingest`.
- Node/Python API shape or type conversion issues: inspect the binding
  `lib.rs`, then the corresponding core engine type.
