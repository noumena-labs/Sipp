# {{PROJECT_NAME}}

A browser-native AI chat application powered by [Sipp](https://github.com/noumena-labs/Sipp).

Requires Node.js 20.19+ or 22.12+ (the range Vite 8 supports).

## Quickstart

```bash
# 1. Install dependencies
npm install

# 2. Start the development server with pre-configured cross-origin isolation
npm run dev
```

Open `http://localhost:5173` in a current browser.

## Features

- **In-Browser Local Inference**: Runs directly in your browser without sending tokens to any external server.
- **GPU When Available**: Uses WebGPU when the browser and GPU support it, and the CPU otherwise.
- **Resumable Hugging Face Downloads**: Downloads models directly into browser OPFS with byte-level range resumes.
- **Cross-Origin Isolation**: Pre-configured COOP (`Cross-Origin-Opener-Policy: same-origin`) and COEP (`Cross-Origin-Embedder-Policy: require-corp`) headers in `vite.config.ts`.

## Scripts

- `npm run dev` - Start Vite dev server
- `npm run build` - Build production bundle
- `npm run preview` - Preview production build
