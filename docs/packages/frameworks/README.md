# Frameworks

These guides show how to use the JavaScript-facing CogentLM packages in common
application frameworks.

Use the browser package, `cogentlm`, when inference runs in the browser or when
browser code calls a gateway. Use the Node package, `cogentlm-server`, only in
server-only code such as route handlers, server functions, API routes, workers,
or services that run in a Node.js runtime.

## Guides

- [Next.js](nextjs.md): App Router route handlers, Client Components,
  gateway proxies, and streaming.
- [TanStack](tanstack.md): TanStack Start server functions and TanStack Query
  patterns.
- [React And Vite](vite-react.md): Baseline browser package setup, WASM asset
  behavior, OPFS model loading, and gateway examples.

## Package Selection

| Environment | Package | Notes |
| --- | --- | --- |
| Browser component | `cogentlm` | Use for browser-local GGUF inference or direct gateway calls. |
| Node server route | `cogentlm-server` | Use only in Node runtime code; do not bundle into browser code. |
| Gateway proxy | Either | Browser code can call the gateway directly with short-lived tokens, or server code can proxy calls with server-held secrets. |

Keep provider credentials and long-lived gateway tokens out of browser bundles.
When a browser app needs gateway access, issue short-lived application tokens or
proxy through a server route.
