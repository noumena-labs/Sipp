# {{PROJECT_NAME}}

A browser-native AI chat application powered by [Sipp](https://github.com/noumena-labs/Sipp) and WebGPU.

Requires Node.js 20.19 or newer.

## Quickstart

```bash
# 1. Install dependencies
npm install

# 2. Start the development server with pre-configured cross-origin isolation
npm run dev
```

Open `http://localhost:5173` in a browser with WebGPU support.

## Features

- **In-Browser Local Inference**: Runs directly in your browser without sending tokens to any external server.
- **WebGPU Acceleration**: Blazing fast inference powered by modern GPU shaders.
- **Resumable Hugging Face Downloads**: Downloads models directly into browser OPFS with byte-level range resumes.
- **Cross-Origin Isolation**: Pre-configured COOP (`Cross-Origin-Opener-Policy: same-origin`) and COEP (`Cross-Origin-Embedder-Policy: require-corp`) headers in `vite.config.ts`.

## Scripts

- `npm run dev` - Start Vite dev server
- `npm run build` - Build production bundle
- `npm run preview` - Preview production build
