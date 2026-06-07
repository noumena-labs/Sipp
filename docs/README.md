# CogentLM Documentation

CogentLM packages local and gateway-backed inference runtimes for browser,
Node.js, Python, and Rust applications. The project is organized around one
client model: register endpoints with `CogentClient.add`, keep the returned
endpoint reference, and choose that reference for `query`, `chat`, or `embed`.

This book is the public documentation for the repository. It starts with
source-based development workflows and documents the public package names used
by the browser, Node.js, Python, Rust, and gateway surfaces.

## Start Here

- [Installation](getting-started/installation.md) explains repository setup and
  source builds.
- [Quickstarts](getting-started/quickstarts.md) shows the shortest local,
  gateway, and browser paths.
- [Packages](packages/browser.md) describes the public package surfaces.
- [Gateway And Hybrid Inference](guides/gateway-hybrid.md) explains when to use
  local endpoints, gateway endpoints, and provider endpoints.
- [Examples And Demos](examples-demos.md) maps the runnable examples and demos.

## Build The Book Locally

The documentation uses mdBook pinned to version `0.5.3`.

```bash
cargo install mdbook --version 0.5.3 --locked
mdbook build
mdbook serve --open
```

`mdbook build` writes generated output to `book/`. GitHub Pages builds the same
book from `.github/workflows/docs.yml`.

## Documentation Ownership

First-party README files stay short and point into this book for deeper guides.
Vendored README files under `third_party/` are upstream-owned.
