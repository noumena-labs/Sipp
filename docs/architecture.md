# Architecture

CogentLM separates inference primitives from protocol and deployment policy.
The public package surfaces compose lower-level crates without moving HTTP
routes, serialized wire formats, or deployment defaults into core inference
layers.

## Published Crates

- `crates/cogentlm`: the public `cogentlm` Rust library. Former foundational
  crates continue as module folders:
  - `core`: low-level shared types.
  - `shard`: GGUF cache planning and split-file utilities.
  - `backend`, `engine`, `lifecycle`, `runtime`: local inference, scheduling,
    lifecycle, and memory management.
  - `client`: typed endpoint registration and query, chat, embed dispatch,
    re-exported at the crate root.
  - `providers` (feature `providers`): explicitly selected external provider
    adapters.
  - `gateway_core` (feature `gateway`): protocol-neutral gateway execution
    traits and pipeline ordering.
- `crates/sys`: the `cogentlm-sys` crate — unsafe FFI bindings, native
  llama.cpp shims, and the vendored `llama.cpp/` source tree.

## Public Libraries

- `lib/web`: browser package source.
- `lib/node`: Node.js server package source.
- `lib/python`: Python package source.
- `lib/gateway`: route-free HTTP gateway toolkit, consumed from source
  checkouts.

## Applications And Examples

- `apps/gateway-server`: opinionated first-party gateway application.
- `apps/cli`: command-line local inference application.
- `examples`: small copyable integrations.
- `demos`: browser experiences built on the public package surfaces.
- `xtask`: build, test, run, packaging, and maintenance orchestration.

For gateway-specific layering, read
[Gateway Architecture](gateway/architecture.md).
