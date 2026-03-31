# Three.js Demo

This demo now doubles as the browser benchmark harness for `cogent-engine`.
It runs the browser-hosted inference path and reports runtime init, model load,
engine init, cold prompt, hot fresh-context, and hot reused-context timings.

## Run

From monorepo root (`cogent-engine/`):

```bash
bun run demo:install
bun run demo:dev
```

Open the Vite URL, load a `.gguf` model from file or URL, then either:

- run `Init Runtime` and `Load Model + Init Engine` for manual single-prompt testing
- run `Run Full Browser Benchmark` for a fresh end-to-end benchmark report

The benchmark panel also shows browser WebGPU availability and lets you export the
raw report as JSON.
