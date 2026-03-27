# cogent-engine monorepo

Standalone monorepo for the `cogent-engine` npm package and the Three.js demo.

## Workspace layout

- `packages/cogent-engine`: npm package and native/WebAssembly bridge
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
npm install
```

## Build package

```bash
npm run build
```

## Rebuild package from clean state

```bash
npm run rebuild:package
```

## Run demo

```bash
npm run demo:install
npm run demo:dev
```

`demo:dev` automatically builds `packages/cogent-engine` first.
