# CLI

`apps/cli` builds the `sipp` command-line application for local GGUF text and
speech inference. It is useful for manual model checks and quick local
requests.

## Build

```bash
cargo xtask build cli --backend cpu
cargo xtask build cli --backend all
```

## Run

```bash
cargo run -p sipp-cli -- <model.gguf> "Explain Sipp."
cargo run -p sipp-cli -- <asr.gguf> --listen <audio.mp3> --projector <asr-mmproj.gguf>
cargo run -p sipp-cli -- <tts.gguf> "Hello" --speak output.wav --projector <tts-mmproj.gguf>
```

Useful flags include:

- `--max-tokens`
- `--ctx-size`
- `--backend auto|cpu|cuda|metal|vulkan`
- `--temperature`
- `--stats off|basic|profile`
- `--chat`
- `--listen <audio>` or `--speak <output.wav>`
- `--projector <mmproj.gguf>`
- `--language` and `--speaker <reference-audio>`

Use `cargo run -p sipp-cli -- --help` for the full generated help.
