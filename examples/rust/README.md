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

Start a gateway first, then set:

```bash
export COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
export COGENTLM_GATEWAY_TOKEN="dev-token"
```

Run:

```bash
cargo run -p cogentlm-rust-examples --features remote --bin gateway_query -- <model.gguf> local [input]
cargo run -p cogentlm-rust-examples --features remote --bin gateway_chat -- <model.gguf> local [input]
cargo run -p cogentlm-rust-examples --features remote --bin gateway_embed -- <model.gguf> local-embed [input]
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
