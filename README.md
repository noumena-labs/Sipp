# cogent-engine monorepo

Standalone monorepo for the `cogent-engine` package and the browser benchmark app.

## Workspace layout

- `packages/cogent-engine`: package and native/WebAssembly bridge
- `packages/cogent-engine/third_party/llama.cpp`: pinned `llama.cpp` submodule
- `apps/benchmark`: browser benchmark app

## Clone

Clone with submodules so the vendored `llama.cpp` checkout is present from the start:

```bash
git clone --recurse-submodules <repo-url> cogent-engine
cd cogent-engine
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

## Install

```bash
bun install
```

## Build package

```bash
bun run build
```

## Rebuild package from clean state

```bash
bun run rebuild:package
```

## Benchmark Inference

```bash
bun run bench:bun --model ./Qwen3.5-0.8B-Q4_0.gguf --json ./benchmarks/latest-bun.json
```

This benchmarks the Bun-hosted runtime path with the standard matrix: model file read, WASM module init, model load into MEMFS, engine init, cold prompts, hot fresh-context prompts, and hot reused-context prompts. Use `--preset single --prompt "..." --tokens 32` for one custom prompt.
The default matrix covers `SISO`, `SILO`, `LISO`, and `LILO`. Headline metrics are reported in a TensorRT-style serving format: `TTFT`, `TPOT`, `ITL`, `E2EL`, request throughput, output token throughput, and total token throughput.

## Run Browser Benchmark App

```bash
bun install
bun run benchmark:dev
```

`benchmark:dev` automatically builds `packages/cogent-engine` first and then starts the
browser benchmark app for the real browser-hosted WebGPU inference path.

## Automated Browser Benchmark

```bash
bun run bench:browser --model ./Qwen3.5-0.8B-Q4_0.gguf --browser chrome --output ./benchmarks/browser/latest.json
```

This launches the benchmark app in a real Chromium browser through Playwright, uploads the
local GGUF model, runs the browser benchmark, and saves the JSON report with browser adapter
and runtime backend metadata.
