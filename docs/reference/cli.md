# CLI

`apps/cli` builds the `cogentlm` command-line application for local GGUF text
generation. It is useful for runtime smoke testing, manual model checks, and
quick local prompts.

## Build

```bash
cargo xtask build cli --backend cpu
cargo xtask build cli --backend all
```

## Run

```bash
cargo run -p cogentlm-cli -- <model.gguf> "Explain CogentLM."
```

Useful flags include:

- `--max-tokens`
- `--ctx-size`
- `--backend auto|cpu|cuda|metal|vulkan`
- `--temperature`
- `--stats off|basic|profile`
- `--chat`

Use `cargo run -p cogentlm-cli -- --help` for the full generated help.
