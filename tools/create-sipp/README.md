# @sipphq/create-sipp

The official browser-app initializer for [Sipp](https://github.com/noumena-labs/Sipp).
It creates a Vite application that runs `@sipphq/sipp` in the browser; it does
not install or run the Node-only `@sipphq/sipp-server` package.

Requires Node.js 20.19 or newer.

## Quickstart

```bash
npm create @sipphq/sipp@latest my-ai-app
# or
npx @sipphq/create-sipp@latest my-ai-app
```

Then follow the on-screen instructions:

```bash
cd my-ai-app
npm install
npm run dev
```

## What's Included

- **Vite + TypeScript**: Fast modern development environment.
- **Cross-Origin Isolation Pre-Wired**: `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers configured in `vite.config.ts` for WebGPU and multithreaded WASM support.
- **Streaming Chat UI**: Complete, responsive, zero-dependency browser chat interface.
- **Resumable Hugging Face Downloads**: Download models directly inside the browser using byte ranges and OPFS caching.
