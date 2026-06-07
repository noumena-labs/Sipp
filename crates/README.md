# Foundational Crates

`crates/` contains the foundational Rust implementation. These crates provide
inference primitives, endpoint registration, gateway execution traits, provider
adapters, and native integration layers used by the public packages.

## Crates

- `sys`: unsafe FFI bindings and native llama.cpp shims.
- `core`: low-level shared types.
- `engine`: local inference, scheduling, model lifecycle, and memory
  management.
- `shard`: GGUF cache planning and split-file utilities.
- `client`: typed endpoint registration and query, chat, embed dispatch.
- `gateway-core`: protocol-neutral gateway execution traits and pipeline
  ordering.
- `providers`: explicitly selected external provider adapters.

## Boundaries

HTTP routes, JSON/SSE contracts, authentication schemes, configuration files,
application limits, and deployment defaults belong in `lib/gateway`,
`apps/gateway-server`, or application code.

See [../docs/architecture.md](../docs/architecture.md) for the public
architecture overview.
