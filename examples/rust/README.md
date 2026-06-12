# Rust Examples

Each file in `src/` is a focused Rust tutorial. `support.rs` only handles
argument parsing, environment helpers, and output formatting.

## Local GGUF

```bash
cargo run -p sipp-rust-examples --bin query -- <model.gguf> [input]
cargo run -p sipp-rust-examples --bin chat -- <model.gguf> [input]
cargo run -p sipp-rust-examples --bin embed -- <model.gguf> [input]
cargo run -p sipp-rust-examples --bin vision_chat -- <model.gguf> <projector.gguf> <image> [input]
```

## Gateway Clients

Use the one-command gateway workflow when possible:

```bash
cargo xtask run examples gateway rust --case query
```

For a manually started gateway, set `SIPP_GATEWAY_URL` and
`SIPP_GATEWAY_TOKEN`, then run:

```bash
cargo run -p sipp-rust-examples --features gateway --bin gateway_query -- <model.gguf> local [input]
cargo run -p sipp-rust-examples --features gateway --bin gateway_chat -- <model.gguf> local [input]
cargo run -p sipp-rust-examples --features gateway --bin gateway_embed -- <model.gguf> local [input]
```

`gateway_embed` requires a model/runtime that reports embedding support.

## Direct Provider Chat

Direct provider examples call the selected provider from the Rust process
without a gateway. By default they use the `gemini` preset, which maps to
Sipp's OpenAI-compatible provider descriptor.

```bash
export SIPP_PROVIDER="gemini"
export GEMINI_API_KEY="<gemini-api-key>"
cargo run -p sipp-rust-examples --bin provider_chat -- [input]
```

For any OpenAI-compatible provider, pass the generic descriptor fields:

```bash
export SIPP_PROVIDER="openai_compatible"
export SIPP_PROVIDER_BASE_URL="https://provider.example/v1"
export SIPP_PROVIDER_API_KEY="<provider-api-key>"
export SIPP_PROVIDER_MODEL="<provider-model>"
cargo run -p sipp-rust-examples --bin provider_chat -- [input]
```

Provider credentials belong in trusted server or gateway processes, not
distributed app clients.

See [../README.md](../README.md) for shared gateway setup details.
