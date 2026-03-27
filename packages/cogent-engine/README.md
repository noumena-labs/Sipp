# cogent-engine

`cogent-engine` is a browser-focused npm package that compiles CogentEngine C++ + `llama.cpp` to WebAssembly and exposes a typed TypeScript API.

Source layout in this package:

- `native/` C++ bridge/manager/wasm exports
- `src/` TypeScript runtime wrapper
- `third_party/llama.cpp/` vendored `llama.cpp`
- `scripts/` build and clean scripts
- `cmake/` shared Emscripten configuration

## Prerequisites

- Node.js 20+ and npm
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

From `packages/cogent-engine/`:

```bash
npm install
npm run build
```

For a full clean rebuild:

```bash
npm run rebuild
```

What this does:

- `npm run build:wasm` compiles `native/` + `third_party/llama.cpp` with Emscripten and writes runtime artifacts to `dist/wasm`
- `npm run build:ts` compiles TypeScript wrapper code to `dist/esm` and declarations to `dist/types`
- `build:wasm` auto-selects a CMake generator, or you can force one via `CMAKE_GENERATOR`

Why first builds are slow:

- Emscripten downloads and caches ports (for example `emdawnwebgpu`) on first use.
- Emscripten also compiles and caches system libraries (`libc`, `libc++`, `libhtml5`, etc.).
- After cache warmup, rebuilds are significantly faster unless you clear the Emscripten cache.

```bash
# example
CMAKE_GENERATOR=Ninja npm run build:wasm
```

```powershell
# Windows PowerShell example with explicit Ninja path
$env:CMAKE_GENERATOR="Ninja"
$env:CMAKE_MAKE_PROGRAM="C:\\Users\\<you>\\Documents\\emsdk\\ninja\\<version>\\ninja.exe"
npm run build:wasm
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
npm run build
```

```bash
# macOS / Linux
rm -rf build dist/wasm
npm run build
```

Or use the package script:

```bash
npm run clean
npm run build
```

## Faster Iteration

- TS-only changes: `npm run build:ts`
- C++/WASM changes: `npm run build:wasm`

## How To Use In An App

Install from local path during development:

```bash
npm install ../packages/cogent-engine
```

Then import and run:

```ts
import { CogentEngine, getBundledRuntimeUrls } from "cogent-engine";

const engine = new CogentEngine(getBundledRuntimeUrls());
await engine.initModule();

const modelPath = await engine.loadModelFromUrl("/models/model.gguf");
await engine.initEngine(modelPath);

const response = await engine.prompt("demo", "Say hello in one sentence.", 64);
console.log(response);
```

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

## Three.js Demo

A Vite + Three.js demo lives in `../../apps/threejs`.

```bash
cd ../../
npm run build
npm run demo:install
npm run demo:dev
```

Open the Vite URL, click runtime init, load a local or remote `.gguf` model, then run inference.
