# Browser Benchmark App

This app is the browser benchmark harness for `@noumena-labs/cogentlm`.
It is intentionally benchmark-focused: no Three.js scene, no decorative WebGL layer,
and no unrelated demo behavior.
Benchmark runs explicitly enable runtime observability and backend profiling so the
exported report can include serving metrics, transport observability, and backend summaries.

It supports:

- manual browser benchmarking through the UI
- JSON report export
- automation through `window.__cogentBench`

## Run

From the monorepo root:

```bash
bun run benchmark:dev
```

`benchmark:dev` builds the browser WebGPU ingest package first: wasm32 WebGPU
with the Rust GGUF ingest splitter linked by Emscripten. Large monolithic GGUF
files are split into OPFS-backed shards on the browser path.

For a production build:

```bash
bun run benchmark:build
```

## Automation

The page exposes a stable automation API:

- `window.__cogentBench.getEnvironment()`
- `window.__cogentBench.getRuntimeObservability()`
- `window.__cogentBench.getBackendObservability()`
- `window.__cogentBench.getBrowserRuntimeSmoke()`
- `window.__cogentBench.runBrowserRuntimeSmoke()`
- `window.__cogentBench.getLastReport()`

This API is used by the automated browser benchmark runner.
