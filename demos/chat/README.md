# CogentLM Chat

A focused browser chat interface for testing local GGUF models with CogentLM
and WebGPU.

## Run

```bash
cargo xtask run demos serve chat
```

Open the printed local URL, choose a curated text or vision model, and load it.
Custom URL and local file imports support text GGUF models.

The interface streams generated tokens and reports request-level decode speed,
time to first token, and output token count.
