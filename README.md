# CogentLM monorepo

Monorepo for the published `@noumena-labs/cogent-engine` package plus the local avatar,
benchmark, and simulation apps that exercise it.

## Workspace layout

- `packages/cogent-engine`: publishable npm package and native/WebAssembly bridge
- `packages/cogent-engine/third_party/llama.cpp`: pinned `llama.cpp` submodule
- `apps/avatar`: browser character harness with a VRM avatar
- `apps/benchmark`: browser benchmark harness
- `apps/simulation`: browser simulation and orchestrator example

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

## Build the package

```bash
bun run build:package
```

Use the release build when you need the publishable package layout with browser and Bun wasm
artifacts:

```bash
bun run build:package:release
```

For a clean rebuild:

```bash
bun run rebuild:package
```

## Publish to GitHub Packages

The repository `.npmrc` already maps `@noumena-labs` to GitHub Packages. Authenticate with either
`NODE_AUTH_TOKEN` or `npm login --scope=@noumena-labs --registry=https://npm.pkg.github.com`, then
run the shared publish helper from the repo root:

```bash
bun run publish:cogent-engine:dry-run
bun run publish:cogent-engine
```

The helper verifies registry configuration and auth, runs the package `release:prepare` flow,
validates the tarball with `npm pack --dry-run`, and then publishes with `npm publish`. The GitHub
Actions workflow at `.github/workflows/publish-cogent-engine.yml` calls the same helper.

## Consume from another private app

Add this to the consuming repository `.npmrc`:

```ini
@noumena-labs:registry=https://npm.pkg.github.com
```

Authenticate with a PAT that has `read:packages` for installs and `write:packages` for publishes,
or use `npm login` against the GitHub Packages registry. Install a pinned version:

```bash
npm install @noumena-labs/cogent-engine@1.0.0
```

Available public imports:

- `@noumena-labs/cogent-engine`
- `@noumena-labs/cogent-engine/character`
- `@noumena-labs/cogent-engine/orchestrator`

Browser apps that use the wasm runtime need `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` so `SharedArrayBuffer` stays available.

## Benchmark inference

```bash
bun run bench:bun --model ./Qwen3.5-0.8B-Q4_0.gguf --json ./benchmarks/latest-bun.json
```

This benchmarks the Bun-hosted runtime path with the standard matrix: model file read, WASM module
init, model load into MEMFS, engine init, cold prompts, hot fresh-context prompts, and hot
reused-context prompts. Use `--preset single --prompt "..." --tokens 32` for one custom prompt.
The default matrix covers `SISO`, `SILO`, `LISO`, and `LILO`. Headline metrics are reported in a
TensorRT-style serving format: `TTFT`, `TPOT`, `ITL`, `E2EL`, request throughput, output token
throughput, and total token throughput.

## Run Browser Benchmark App

```bash
bun run benchmark:dev
```

`benchmark:dev` automatically builds `packages/cogent-engine` first and then starts the browser
benchmark app for the real browser-hosted WebGPU inference path.

## Automated Browser Benchmark

```bash
bun run bench:browser --model ./Qwen3.5-0.8B-Q4_0.gguf --browser chrome --output ./benchmarks/browser/latest.json
```

This launches the benchmark app in a real Chromium browser through Playwright, uploads the local
GGUF model, runs the browser benchmark, and saves the JSON report with browser adapter and runtime
backend metadata.
