# cogent-engine

`cogent-engine` is a browser-focused package that compiles an inference-only CogentEngine C++ runtime plus `llama.cpp` to WebAssembly and exposes a typed TypeScript API.

Source layout in this package:

- `native/` C++ inference runtime, bridge, and wasm exports
- `src/` TypeScript runtime wrapper
- `third_party/llama.cpp/` vendored `llama.cpp`
- `scripts/` build and clean scripts
- `cmake/` shared Emscripten configuration

## Design Docs

- `docs/inference-runtime-v2-design.md` -> detailed target architecture, data structures, algorithms, implementation phases, and reference bibliography
- `docs/inference-runtime-v2-implementation-guide.md` -> concrete execution checklist, file targets, verification gates, and per-phase working order
- `docs/phase-1-implementation-workplan.md` -> step-by-step manual Phase 1 handoff with exact function order, references, and "what next" guidance
- `docs/inference-architecture-draft.md` -> short overview that now points to the detailed design

## Prerequisites

- Bun 1.3+
- Emscripten SDK (`emcmake`, `emcc`) in `PATH`
- CMake 3.20+ and at least one CMake generator tool (`ninja`, `nmake`, or `make`)

If Emscripten is installed but not active in your shell, activate it first (example):

```bash
# macOS / Linux
source /path/to/emsdk/emsdk_env.sh

# Windows PowerShell
/path/to/emsdk/emsdk_env.ps1
```

## Complete Compile (From Source)

From the monorepo root:

```bash
bun install
bun run build:package
```

Or from `packages/cogent-engine/` after the workspace has already been installed:

```bash
bun run build
```

For a full clean rebuild:

```bash
bun run rebuild
```

What this does:

- `bun run build:wasm` compiles `native/` + `third_party/llama.cpp` with Emscripten and writes runtime artifacts to `dist/wasm`
- `bun run build:ts` compiles TypeScript wrapper code to `dist/esm` and declarations to `dist/types`
- `build:wasm` auto-selects a CMake generator, or you can force one via `CMAKE_GENERATOR`

Why first builds are slow:

- Emscripten downloads and caches ports (for example `emdawnwebgpu`) on first use.
- Emscripten also compiles and caches system libraries (`libc`, `libc++`, `libhtml5`, etc.).
- After cache warmup, rebuilds are significantly faster unless you clear the Emscripten cache.

```bash
# example
CMAKE_GENERATOR=Ninja bun run build:wasm
```

```powershell
# Windows PowerShell example with explicit Ninja path
$env:CMAKE_GENERATOR="Ninja"
$env:CMAKE_MAKE_PROGRAM="C:\\Users\\<you>\\Documents\\emsdk\\ninja\\<version>\\ninja.exe"
bun run build:wasm
```

Build outputs:

- `dist/esm` -> JS API entrypoints
- `dist/types` -> `.d.ts` files
- `dist/wasm` -> `cogent-engine-wasm.js` + `cogent-engine-wasm.wasm`

## Clean Rebuild

Use this when you want a full rebuild from scratch:

```bash
# Windows PowerShell
Remove-Item -Recurse -Force build, dist\wasm -ErrorAction SilentlyContinue
bun run build
```

```bash
# macOS / Linux
rm -rf build dist/wasm
bun run build
```

Or use the package script:

```bash
bun run clean
bun run build
```

## Faster Iteration

- TS-only changes: `bun run build:ts`
- C++/WASM changes: `bun run build:wasm`

## Bun Benchmark

From `packages/cogent-engine/`:

```bash
bun run bench:bun --model ../../Qwen3.5-0.8B-Q4_0.gguf --json ./benchmarks/latest-bun.json
```

The benchmark measures:

- model file read time
- WASM module initialization
- model copy into MEMFS
- engine initialization
- TTFT from the first streamed token callback
- TPOT from `(E2EL - TTFT) / (output_tokens - 1)`
- ITL from token-to-token callback intervals
- E2EL from request start to final streamed token completion
- request throughput, output token throughput, and total token throughput
- cold prompt latency
- hot prompt latency with fresh contexts
- hot prompt latency with a reused context

It also reports native `llama.cpp` perf counters for prompt eval, decode eval, and sampling, and saves a structured report with:

- benchmark preset and scenario metadata
- artifact label and inferred quantization label
- init config and prompt format
- Bun runtime metadata
- per-scenario cold, hot fresh-context, and hot reused-context groups
- SISO, SILO, LISO, and LILO as the default matrix
- TensorRT-style serving metrics as the primary summary
- prompt-eval and decode throughput as secondary runtime diagnostics
- logical input tokens and effective prompt-eval tokens reported separately

By default, `bench:bun` runs the standard matrix. Use `--preset single --prompt "..." --tokens 32` when you want one custom prompt instead.

You can also sweep the Phase 1 init config directly from the benchmark:

```bash
bun run bench:bun \
  --model ../../Qwen3.5-0.8B-Q4_0.gguf \
  --ctx 4096 \
  --batch 256 \
  --ubatch 256 \
  --threads 4 \
  --threads-batch 4 \
  --gpu-layers 99 \
  --flash-attention auto \
  --kv-unified true
```

## WebGPU Backend-Ops Runner

The package includes a browser-hosted WebGPU runner for the vendored `llama.cpp` `test-backend-ops` target.

Install Chromium for Playwright once before using these commands:

```bash
bunx playwright install chromium
```

From the monorepo root:

```bash
bun run test:backend-ops:webgpu -- --list-ops
bun run test:backend-ops:webgpu:op -- GET_ROWS
bun run test:backend-ops:webgpu:op -- GET_ROWS,SET_ROWS --mode support --output csv
bun run test:backend-ops:webgpu:op -- "Get Rows" --filter "type=f32"
```

What the op wrapper does:

- maps friendly op names like `get-rows`, `Get Rows`, or `GET_ROWS` onto the upstream `-o GET_ROWS` selector
- defaults to WebGPU by relying on the underlying runner's automatic `-b WebGPU` injection
- forwards modes to upstream `test-backend-ops`: `test`, `support`, `perf`, and `grad`
- forwards parameter regex filtering through `-p`

If you need the full upstream CLI surface, keep using the raw passthrough command:

```bash
bun run test:backend-ops:webgpu -- test -o MUL_MAT -p "type=f16"
```

## Debugging WebGPU Backend-Ops Wasm

Use the debug wrapper to build `test-backend-ops` in `Debug` with `CE_WASM_DEBUG=ON` and bundled DWARF symbols inside the generated wasm:

```bash
bun run test:backend-ops:webgpu:debug -- GET_ROWS
```

The debug command:

- uses a dedicated build directory: `build-test-backend-ops-webgpu-debug`
- configures CMake with `CMAKE_BUILD_TYPE=Debug`
- enables debugger-friendly Emscripten flags such as `-g3` and assertions
- normalizes Windows source-path drive-letter casing in DWARF to improve VS Code breakpoint binding
- launches headed Chromium on remote debug port `9222`
- pauses before `callMain()` so you can attach a debugger first

Before using C++ breakpoints in VS Code, install the `WebAssembly DWARF Debugging` extension: `ms-vscode.wasm-dwarf-debugging`.

VS Code workflow from the repo root:

1. Run the `Attach backend-ops WebGPU Debug` launch configuration.
2. Wait for Chromium to open the runner page and pause.
3. Set breakpoints in the wasm-backed sources you want to inspect.
4. Click `Resume Wasm Run` in the browser page to start the selected backend-op run.

If a C++ breakpoint stays gray-hollow in VS Code on Windows, rebuild the debug target after these settings changes so the wasm DWARF paths are regenerated with canonical drive-letter casing.

The repository also includes `.vscode/tasks.json` and `.vscode/launch.json` to automate this attach flow.

## How To Use In An App

Install from local path during development:

```bash
bun add ../packages/cogent-engine
```

Then import and run:

```ts
import { CogentEngine, getBundledRuntimeUrls } from "cogent-engine";

const engine = new CogentEngine(getBundledRuntimeUrls());
await engine.initModule();

const modelPath = await engine.loadModelFromUrl("/models/model.gguf");
await engine.initEngine(modelPath, {
  nCtx: 4096,
  nBatch: 256,
  nUbatch: 256,
  nGpuLayers: 99,
  flashAttention: "auto",
});

const response = await engine.streamPrompt("demo", "Say hello in one sentence.", {
  nTokens: 64,
  onToken: (token) => {
    process.stdout.write(token);
  },
});
console.log(response);
```

`prompt()` still exists as the convenience wrapper over the streaming path when you only want the final string.

`getBundledRuntimeUrls()` is the clean default when you want to use the runtime assets packaged with `cogent-engine`.

`moduleUrl` and `wasmUrl` are still available for advanced cases where you want to host the runtime assets somewhere else.
By default, only same-origin module/wasm URLs are allowed.

If you host wasm assets on CDN/static storage:

```ts
const engine = new CogentEngine({
  moduleUrl: "https://cdn.example.com/cogent-engine-wasm.js",
  wasmUrl: "https://cdn.example.com/cogent-engine-wasm.wasm",
  trustedOrigins: ["https://cdn.example.com"],
});
```

`loadModelFromUrl()` requires a valid `Content-Length` header by default. If your host cannot provide one, enable it explicitly:

```ts
const engine = new CogentEngine({
  ...getBundledRuntimeUrls(),
  allowUnknownContentLength: true,
});
```

## Browser Benchmark App

A browser benchmark app lives in `../../apps/benchmark`.

```bash
cd ../../
bun install
bun run benchmark:dev
```

Open the Vite URL, initialize the runtime, load a local or remote `.gguf` model, then run the browser benchmark.
