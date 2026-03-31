# cogent-engine monorepo

Standalone monorepo for the `cogent-engine` package and the Three.js demo.

## Workspace layout

- `packages/cogent-engine`: package and native/WebAssembly bridge
- `packages/cogent-engine/third_party/llama.cpp`: pinned `llama.cpp` submodule
- `apps/threejs`: Three.js demo app

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
bun run bench:bun --model ./Qwen3.5-0.8B-Q4_0.gguf --tokens 16 --warmup 1 --runs 3
```

This benchmarks the Bun-hosted runtime path: model file read, WASM module init, model load into MEMFS, engine init, first prompt, hot fresh-context prompts, and hot reused-context prompts.

## Run demo

```bash
bun run demo:install
bun run demo:dev
```

`demo:dev` automatically builds `packages/cogent-engine` first. The Three.js app now
includes the browser benchmark harness for the real browser-hosted inference path.
