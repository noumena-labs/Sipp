# Sipp Architecture Guide

Sipp separates inference primitives from protocol and deployment policy.

## Published Crates

- **`crates/sipp`**: The public `sipp-rs` package published to crates.io with
  library crate name `sipp`.
  Internal module folders keep the former crate boundaries:
  - **`core/`**: Low-level shared types (`sipp::core`).
  - **`shard/`**: GGUF cache planning and split-file utilities (`sipp::shard`).
  - **`backend/`, `engine/`, `lifecycle/`, `runtime/`**: Local inference, scheduling, model lifecycle, and memory management (former engine crate, public paths unchanged).
  - **`client/`**: Typed query, chat, and embed dispatch re-exported at the crate root. It owns local, provider, and gateway endpoint descriptors.
  - **`providers/`** (feature `providers`): Explicitly selected external provider adapters (`sipp::providers`).
  - **`gateway_core/`** (feature `gateway`): Protocol-neutral gateway execution (`sipp::gateway_core`). It owns request context, cancellation, typed pipeline ordering, streaming events, and resolver, authorizer, admission, and executor traits.
- **`crates/sys`**: The `sipp-sys` crate — unsafe FFI bindings, native llama.cpp shims, and the vendored `llama.cpp/` source tree.

Nothing under `crates/` owns HTTP routes, JSON/SSE contracts, authentication schemes, configuration files, application limits, or deployment defaults.

## Developer Libraries

- **`lib/gateway`**: Route-free HTTP gateway toolkit outside the published crates (`publish = false`, consumed from source checkouts). It provides codecs, authentication traits, error translation, observability, and response helpers.
- **`lib/gateway::GatewayCodec`**: Optional first-party query/chat/embed JSON and SSE profile.
- **`sipp::GatewayEndpointConfig`**: Client-owned descriptor for calling an HTTP gateway endpoint.
- **`lib/node`**, **`lib/python`**, and **`lib/web`**: Public language packages exposing local inference, provider adapters, and gateway endpoint descriptors through the unified add API.

Arbitrary wire formats are implemented programmatically through `ProtocolCodec`. The core client remains limited to the query, chat, and embed inference capabilities.

## Applications

- **`apps/gateway-server`**: First-party Axum application. It owns TOML, routes, bearer authentication, target policy, concurrency limits, CORS, metrics, listeners, and deployment behavior.
- **`examples/gateway`**: Minimal Axum composition with application-owned routes that decode requests, call `SippClient` directly, and encode responses.
- **`apps/cli`**: Command-line local inference application.
- **`xtask`**: Build, test, run, and packaging orchestration.

## Bindings

- **`bindings/node`**: N-API host binding.
- **`bindings/python`**: PyO3 host binding.
- **`bindings/wasm`**: Browser WebAssembly/WebGPU ABI and native link target.

Bindings expose endpoint construction but do not move protocol policy back into the client modules.
