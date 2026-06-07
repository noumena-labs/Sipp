# Browser Playground

This tool is the browser playground for `@noumena-labs/cogentlm`.
It exposes runtime diagnostics, browser environment details, backend observability,
and repeatable measurement runs for debugging browser-hosted inference.

It supports:

- manual runtime diagnostics through the UI
- JSON report export
- automation through `window.__cogentPlayground`

## Run

From the monorepo root:

```bash
cargo xtask run tools serve playground
```

`run tools serve playground` builds the browser WebGPU ingest package first:
wasm32 WebGPU with the Rust GGUF ingest splitter linked by Emscripten. Large
monolithic GGUF files are split into OPFS-backed shards on the browser path.

For a production build:

```bash
cargo xtask run tools build playground
```

For the automated playground runtime smoke:

```bash
cargo xtask test smoke suite playground-browser
```

## Automation

The page exposes a stable automation API:

- `window.__cogentPlayground.getEnvironment()`
- `window.__cogentPlayground.getRuntimeObservability()`
- `window.__cogentPlayground.getBackendObservability()`
- `window.__cogentPlayground.getRuntimeSmoke()`
- `window.__cogentPlayground.runRuntimeSmoke()`
- `window.__cogentPlayground.getLastReport()`

This API is used by the automated browser playground smoke runner.
