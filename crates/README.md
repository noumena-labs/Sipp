# Published Crates

`crates/` contains the two crates published to crates.io.

## Crates

- `cogentlm`: the public Rust library. Former foundational crates live on as
  module folders — `core` (shared types), `shard` (GGUF planning and
  split-file utilities), the engine modules (`backend`, `engine`, `lifecycle`,
  `runtime`), the root-level client API, `providers` (feature `providers`),
  and `gateway_core` (feature `gateway`).
- `sys`: unsafe FFI bindings, native llama.cpp shims, and the vendored
  `llama.cpp/` source tree (`cogentlm-sys`).

## Boundaries

HTTP routes, JSON/SSE contracts, authentication schemes, configuration files,
application limits, and deployment defaults belong in `lib/gateway`,
`apps/gateway-server`, or application code.

See [../docs/architecture.md](../docs/architecture.md) for the public
architecture overview.
