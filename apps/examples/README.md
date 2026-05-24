# Cogent Engine Examples

This directory contains minimal, developer-centric examples for the `cogentlm` library.

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

- **Basic Chat**: Simple text-to-text streaming interface.
- **Multimodal Vision**: Guide on how to use vision-language models with `Uint8Array` media.
- **Structured Output**: Using GBNF grammars to extract typed JSON data.
- **Observability**: Real-time performance monitoring (Tokens/sec, TTFT, etc.).
- **Query**: Raw prompt completion through `engine.query()`, including encoder-decoder models.
- **Embeddings**: Vector extraction through `engine.embed()` for embedding-capable models.

## Important Note: COOP/COEP
Cogent Engine requires `SharedArrayBuffer` for multi-threaded WASM execution. The included `vite.config.ts` is configured with the necessary `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers.
