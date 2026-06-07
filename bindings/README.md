# Bindings

`bindings/` contains native and browser ABI layers used by the public language
packages. These crates are implementation boundaries, not the primary package
documentation entry points.

## Directories

- `bindings/node`: N-API host binding used by the Node.js package.
- `bindings/python`: PyO3 host binding used by the Python package.
- `bindings/wasm`: browser WebAssembly/WebGPU ABI and native link target.

## Boundaries

Bindings stay thin over stable core APIs. They expose endpoint
construction and runtime lifecycle behavior without moving protocol policy,
HTTP routes, or deployment defaults out of applications and gateway crates.

See [../docs/architecture.md](../docs/architecture.md) for the architecture
overview and package READMEs under `lib/` for user-facing APIs.
