# CogentLM

CogentLM provides local and gateway-backed inference runtimes for browser,
Node.js, Python, and Rust applications. The project centers on one endpoint
model: register local, gateway, or provider endpoints with `CogentClient.add`,
then choose the returned endpoint reference for `query`, `chat`, or `embed`.

Use this source checkout for builds, examples, demos, and tests. Public package
names are documented below; source package manifests live under `lib/`.

## Packages

| Surface | Public package target | Source |
| --- | --- | --- |
| Browser | `cogentlm` | [`lib/web`](lib/web/README.md) |
| Node.js | `cogentlm-server` | [`lib/node`](lib/node/README.md) |
| Python | `cogentlm` | [`lib/python`](lib/python/README.md) |
| Rust | `cogentlm` | [`lib/rust`](lib/rust/README.md) |
| Gateway toolkit | `cogentlm-gateway` | [`lib/gateway`](lib/gateway/README.md) |
| Gateway server | `cogentlm-gateway-server` | [`apps/gateway-server`](apps/gateway-server/README.md) |

## Start From Source

Use the repository orchestrator for builds, examples, demos, and tests:

```bash
cargo xtask test list
cargo run -p cogentlm-rust-examples --bin query -- <model.gguf> "Explain local inference."
cargo xtask run examples serve browser
cargo xtask run demos serve chat
```

The Rust, Node.js, Python, and browser examples all start with local GGUF
inference. Gateway workflows are available when an application needs a separate
HTTP boundary.

## Documentation

The documentation lives in [`docs`](docs/README.md) and is built with mdBook:

```bash
cargo install mdbook
mdbook build
mdbook serve --open
```

Start with:

- [Installation](docs/getting-started/installation.md)
- [Quickstarts](docs/getting-started/quickstarts.md)
- [Gateway And Hybrid Inference](docs/guides/gateway-hybrid.md)
- [Examples And Demos](docs/examples-demos.md)
- [Testing](docs/testing.md)

## Repository Layout

- [`crates`](crates/README.md): foundational Rust crates.
- [`lib`](lib/rust/README.md): public package facades and gateway toolkit.
- [`bindings`](bindings/README.md): Node, Python, and browser WASM bindings.
- [`apps`](apps/README.md): first-party applications.
- [`examples`](examples/README.md): small, runnable integrations.
- [`demos`](demos/README.md): browser demos built on public package surfaces.
- [`tools/playground`](tools/playground/README.md): browser runtime diagnostics.
- [`xtask`](xtask): build, test, run, and packaging automation.

## Development

Use `cargo xtask` instead of direct build commands when compiling CogentLM
targets. The orchestrator manages native dependencies, backend toolchains, and
package outputs.

Common validation entry points:

```bash
cargo xtask test list
cargo xtask test unit group full
cargo xtask test smoke group examples --backend cpu
```

See [docs/testing.md](docs/testing.md) and [docs/coverage.md](docs/coverage.md)
for the test catalog and coverage workflow.

## License

CogentLM is licensed under Apache-2.0. Vendored third-party components keep
their upstream licenses and documentation.
