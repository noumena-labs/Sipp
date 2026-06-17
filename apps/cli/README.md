# Sipp CLI

`apps/cli` builds the `sipp` command-line application for local GGUF text
generation. It is intended for runtime smoke checks, manual model validation,
and quick local prompts.

## Build

```bash
cargo xtask build cli --backend cpu
cargo xtask build cli --backend all
```

## Run

```bash
cargo run -p sipp-cli -- <model.gguf> "Explain Sipp."
```

Useful flags include:

- `--max-tokens`
- `--ctx-size`
- `--backend auto|cpu|cuda|metal|vulkan`
- `--temperature`
- `--stats off|basic|profile`
- `--chat`

Use `cargo run -p sipp-cli -- --help` for the full generated help.

See [../../docs/en/reference/cli.md](../../docs/en/reference/cli.md) for the CLI
reference.
