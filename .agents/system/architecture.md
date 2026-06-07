# CogentLM Architecture Guide

CogentLM separates inference primitives from protocol and deployment policy.

## Foundational Crates

- **`crates/sys`**: Unsafe FFI bindings and native llama.cpp shims.
- **`crates/core`**: Low-level shared types.
- **`crates/engine`**: Local inference, scheduling, model lifecycle, and memory management.
- **`crates/shard`**: GGUF cache planning and split-file utilities.
- **`crates/client`**: Typed query, chat, and embed dispatch. It owns local, provider, and gateway endpoint descriptors.
- **`crates/gateway-core`**: Protocol-neutral gateway execution. It owns request context, cancellation, typed pipeline ordering, streaming events, and resolver, authorizer, admission, and executor traits.
- **`crates/providers`**: Explicitly selected external provider adapters. Provider wire requirements do not define gateway-core behavior.

Nothing under `crates/` owns HTTP routes, JSON/SSE contracts, authentication schemes, configuration files, application limits, or deployment defaults.

## Developer Libraries

- **`lib/gateway`**: Route-free HTTP gateway toolkit outside the foundational crates. It provides codecs, authentication traits, error translation, observability, and response helpers.
- **`lib/gateway::GatewayCodec`**: Optional first-party query/chat/embed JSON and SSE profile.
- **`cogentlm_client::GatewayEndpointConfig`**: Client-owned descriptor for calling an HTTP gateway endpoint.
- **`lib/rust`**: Public Rust facade.
- **`lib/node`**, **`lib/python`**, and **`lib/web`**: Public language packages exposing local inference, provider adapters, and gateway endpoint descriptors through the unified add API.

Arbitrary wire formats are implemented programmatically through `ProtocolCodec`. The core client remains limited to the query, chat, and embed inference capabilities.

## Applications

- **`apps/gateway-server`**: First-party Axum application. It owns TOML, routes, bearer authentication, target policy, concurrency limits, CORS, metrics, listeners, and deployment behavior.
- **`examples/gateway`**: Minimal Axum composition with application-owned routes that decode requests, call `CogentClient` directly, and encode responses.
- **`apps/cli`**: Command-line local inference application.
- **`xtask`**: Build, test, run, and packaging orchestration.

## Bindings

- **`bindings/node`**: N-API host binding.
- **`bindings/python`**: PyO3 host binding.
- **`bindings/wasm`**: Browser WebAssembly/WebGPU ABI and native link target.

Bindings expose endpoint construction but do not move protocol policy back into `crates/client`.
