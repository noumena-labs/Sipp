# Using Published Packages

CogentLM exposes one endpoint-oriented client model across public package
surfaces. Register a local, gateway, or provider endpoint with
`CogentClient.add`, keep the returned endpoint reference, and pass that
reference to `query`, `chat`, or `embed`.

Most developers should start here instead of building from source.

## Package Surfaces

| Surface | Install | Primary use |
| --- | --- | --- |
| [Browser](browser.md) | `npm install cogentlm` | Browser-local GGUF inference, WebGPU/WASM runtime, and browser gateway clients. |
| [Node.js](node.md) | `npm install cogentlm-server` | Node server processes, route handlers, and backend services. |
| [Python](python.md) | `pip install cogentlm` | Python services, scripts, and gateway clients. |
| [Rust](rust.md) | `cargo add cogentlm` | Rust applications and services. |
| [Gateway Server](gateway-server.md) | Source-built today | First-party HTTP gateway for local and provider targets. |
| [Gateway Toolkit](gateway.md) | Rust source artifact today | Rust toolkit for custom gateway applications. |

The current release workflow publishes browser npm, Node npm, Python wheel,
and Rust source artifacts. The gateway server is documented as a user-facing
deployment surface, but it does not yet have a published binary or public image.

## Common Workflows

- Use a local endpoint when the current browser, server process, Python script,
  or Rust application owns the GGUF model lifecycle.
- Use a gateway endpoint when a separate gateway owns model paths, provider
  credentials, access policy, concurrency, and metrics.
- Use framework guides when integrating the JavaScript packages with
  [Next.js](frameworks/nextjs.md), [TanStack](frameworks/tanstack.md), or
  [React and Vite](frameworks/vite-react.md).
- Use [Source Builds](../maintainers/source-builds.md) when developing the
  repo, staging packages, running demos, or deploying the gateway server from
  this checkout.
