# CogentClient Examples

This directory contains minimal, developer-centric examples for the `cogentlm-browser` library.

## Getting Started

1. **Install Dependencies**:
   ```bash
   bun install
   ```

2. **Run the Dev Server**:
   ```bash
   bun run dev
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5173`.

## Included Examples

- **Basic Chat**: Simple chat interface with interactive token delivery.
- **Multimodal Vision**: Guide on how to use vision-language models with `Uint8Array` media.
- **Structured Output**: Using GBNF grammars to extract typed JSON data.
- **Observability**: Real-time performance monitoring (Tokens/sec, TTFT, etc.).
- **Query**: Raw prompt completion through `client.query()`, including encoder-decoder models and the run-handle response API.
- **Embeddings**: Vector extraction through `client.embed().response` for embedding-capable models.

## Important Note: COOP/COEP
CogentClient requires `SharedArrayBuffer` for multi-threaded WASM execution. The included `vite.config.ts` is configured with the necessary `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers.
