# Using the Core Library

Sipp exposes one endpoint-oriented client model across all public package
surfaces. See the [Library API Overview](../api) for the shared
`SippClient.add`, `query`, `chat`, and `embed` contracts, endpoint descriptor
reference, and gateway-client symmetry patterns.

Most developers should start here instead of building from source.

## Package Surfaces

| Surface | Install | Primary use |
| --- | --- | --- |
| [Library API Overview](../api) | — | Shared `add`, `query`, `chat`, and `embed` contracts across all surfaces. |
| [Browser](browser.md) | `npm install @sipp/sipp` | Browser-local GGUF inference, WebGPU/WASM runtime, and browser gateway clients. |
| [Node.js](node.md) | `npm install @sipp/sipp-server` | Node server processes, route handlers, and backend services. |
| [Python](python.md) | `pip install sipp-py` | Python services, scripts, and gateway clients. |
| [Rust](rust.md) | `cargo add sipp-rs` | Rust applications and services. |
| [Gateway Server](../gateway/server.md) | Source-built today | First-party HTTP gateway for local and provider targets. |
| [Gateway Docker](../gateway/docker.md) | Docker from source | Local and production container workflows for the gateway server. |
| [Gateway Toolkit](../gateway/toolkit.md) | Source-built today | Rust toolkit for custom gateway applications. |

The current release workflow publishes browser npm, Node npm, Python wheels,
and Rust crates. The gateway server is documented in the
[Gateway](../gateway/) section as a user-facing deployment surface, but it does
not yet have a published binary or public image.

## Framework Guides

When integrating JavaScript packages with a framework, see:

- [React and Vite](frameworks/vite-react.md)
- [Next.js](frameworks/nextjs.md)
- [TanStack](frameworks/tanstack.md)

## Supporting Reference

- [Providers](../guides/providers.md) — provider and gateway provider split
- [Runtime Options](../reference/runtime-options.md) — option layer map and
  field reference
- [Source Builds](../maintainers/source-builds.md) — developing from this
  checkout
