# CogentLM Documentation

CogentLM packages local and gateway-backed inference runtimes for browser,
Node.js, Python, and Rust applications. The project is organized around one
client model: register endpoints with `CogentClient.add`, keep the returned
endpoint reference, and choose that reference for `query`, `chat`, or `embed`.

This book starts with the published packages that application developers use.
Source checkout, build orchestration, repository architecture, and contribution
workflow live in the maintainer section.

## Start Here

- [Installation](getting-started/installation.md) lists the published package
  install commands.
- [Quickstarts](getting-started/quickstarts.md) shows short Browser, Node.js,
  Python, Rust, and gateway paths.
- [Using Published Packages](packages/) describes the public package
  surfaces in depth.
- [Gateway Server](packages/gateway-server.md) explains source/exe operation
  for the first-party gateway. [Gateway Server Docker](packages/gateway-server-docker.md)
  covers container deployment.
- [Frameworks](packages/frameworks/) covers Next.js, TanStack, and
  React/Vite integration patterns.
- [Gateway And Hybrid Inference](guides/gateway-hybrid.md) explains when to use
  local endpoints, gateway endpoints, and provider endpoints.
- [Maintainers](maintainers/) covers source builds, tests, repo
  structure, and contribution workflow.

## Build The Book Locally

The documentation uses mdBook pinned to version `0.5.3`.

```bash
cargo install mdbook
mdbook build
mdbook serve --open
```

`mdbook build` writes generated output to `book/`. GitHub Pages builds the same
book from `.github/workflows/docs.yml`.

## Documentation Ownership

First-party README files stay short and point into this book for deeper guides.
Vendored README files under `third_party/` are upstream-owned.
