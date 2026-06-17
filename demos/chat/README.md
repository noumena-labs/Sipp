# Sipp Chat Demo

`demos/chat` is a focused browser chat interface for testing local GGUF models
with Sipp and WebGPU.

## Run

```bash
cargo xtask run demos serve chat
```

Open the printed local URL, choose a curated text or vision model, and load it.
Custom URL and local file imports support text GGUF models.

## What It Demonstrates

- Browser-local model loading.
- Streaming generated tokens.
- Request-level decode speed.
- Time to first token and output token counts.
- Text and vision model selection paths.

See [../../docs/en/examples-demos.md](../../docs/en/examples-demos.md) for the demo
index.
