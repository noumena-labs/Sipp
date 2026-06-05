# CogentLM

CogentLM packages local and gateway-backed inference runtimes for browser, Node.js,
Python, and Rust applications.

## Packages

* `cogentlm`: public browser and Rust package name.
* `@noumena-labs/cogentlm`: internal browser package on GitHub Packages.
* `cogentlm-server`: public Node.js server package name.
* `@noumena-labs/cogentlm-server`: internal Node.js server package on GitHub Packages.
* `cogentlm`: Python package name on PyPI.

## Repository Layout

* `lib/rust`: Rust facade crate.
* `lib/python`: Python package source and packaging metadata.
* `bindings/node`, `bindings/python`, `bindings/wasm`: Rust FFI build code.
* `lib/web`: browser package source for `@noumena-labs/cogentlm`.
* `lib/node`: Node package source for `@noumena-labs/cogentlm-server`.
* `demos`: browser demos, served with `cargo xtask run demos serve chat`.
* `tools/playground`: browser runtime playground and diagnostics tool, served with `cargo xtask run tools serve playground`.
* `examples/node`, `examples/python`, `examples/rust`, `examples/web`: runnable examples; serve browser examples with `cargo xtask run examples serve browser`; see `examples/README.md`.

Use `cargo xtask` commands from this repository to build native artifacts and
language packages. After running `source setup.sh` on Unix-like shells or
`. .\setup.ps1` in PowerShell, use `clm` as the short repo-local alias for
`cargo xtask`.
