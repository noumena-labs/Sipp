# Architecture

CogentLM separates inference primitives from protocol and deployment policy.
The public package surfaces compose lower-level crates without moving HTTP
routes, serialized wire formats, or deployment defaults into core inference
layers.

## Foundational Crates

- `crates/sys`: unsafe FFI bindings and native llama.cpp shims.
- `crates/core`: low-level shared types.
- `crates/engine`: local inference, scheduling, lifecycle, and memory
  management.
- `crates/shard`: GGUF cache planning and split-file utilities.
- `crates/client`: typed endpoint registration and query, chat, embed dispatch.
- `crates/gateway-core`: protocol-neutral gateway execution traits and
  pipeline ordering.
- `crates/providers`: explicitly selected external provider adapters.

## Public Libraries

- `lib/rust`: Rust facade crate.
- `lib/web`: browser package source.
- `lib/node`: Node.js server package source.
- `lib/python`: Python package source.
- `lib/gateway`: route-free HTTP gateway toolkit.

## Applications And Examples

- `apps/gateway-server`: opinionated first-party gateway application.
- `apps/cli`: command-line local inference application.
- `examples`: small copyable integrations.
- `demos`: browser experiences built on the public package surfaces.
- `xtask`: build, test, run, packaging, and maintenance orchestration.

For gateway-specific layering, read
[Gateway Architecture](gateway/architecture.md).
