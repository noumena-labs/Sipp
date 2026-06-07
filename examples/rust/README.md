# Rust Examples

Each file in `src/` is a focused tutorial. `support.rs` only handles argument
parsing, environment helpers, and output formatting.

Endpoints use the unified descriptor API:

```rust
let endpoint = client
    .add("local", EndpointDescriptor::local(model_path, runtime))
    .await?;
```

## Local GGUF

```bash
cargo run -p cogentlm-rust-examples --bin query -- <model.gguf> [input]
cargo run -p cogentlm-rust-examples --bin chat -- <model.gguf> [input]
cargo run -p cogentlm-rust-examples --bin embed -- <model.gguf> [input]
cargo run -p cogentlm-rust-examples --bin vision_chat -- <model.gguf> <projector.gguf> <image> [input]
```

## Gateway Clients

To start the local gateway and run one Rust gateway client from a single
terminal:

```bash
cargo xtask run examples gateway rust --case query
```

The cached sample model under `.build/models` is used by default; pass
`--model <model.gguf>` to override it.

Start a gateway first, then set:

```bash
export COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
export COGENTLM_GATEWAY_TOKEN="dev-token"
```

Run:

```bash
cargo run -p cogentlm-rust-examples --features gateway --bin gateway_query -- <model.gguf> local [input]
cargo run -p cogentlm-rust-examples --features gateway --bin gateway_chat -- <model.gguf> local [input]
cargo run -p cogentlm-rust-examples --features gateway --bin gateway_embed -- <model.gguf> local [input]
```

`gateway_embed` requires a model/runtime that reports embedding support.

## OpenAI Provider

These examples require `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY="<openai-api-key>"
cargo run -p cogentlm-rust-examples --bin openai_provider_chat -- [input]
```

`openai_provider_chat.rs` shows direct provider inference. Provider credentials
belong in gateway/server processes, not distributed app clients.
