# Minimal Gateway Example

This example is the smallest Axum proxy built from `cogentlm-gateway`.
It deliberately has no authentication, CORS, TOML configuration, dashboard,
history, or deployment policy.

It demonstrates:

1. Loading one local GGUF model into `CogentClient`.
2. Publishing it as the `local` gateway alias.
3. Mounting `/v1/query`, `/v1/chat`, and `/v1/embed`.
4. Returning finite JSON or SSE responses.
5. Stopping on Ctrl-C.

Run it with:

```bash
cargo xtask run examples serve gateway-local \
  --model .build/models/model.gguf \
  --bind 127.0.0.1:8787
```

Or directly:

```bash
cargo run -p cogentlm-gateway-example -- \
  --model path/to/model.gguf \
  --bind 127.0.0.1:8787
```

The page served at `/` is stored in `assets/index.html`; no HTML is embedded in
the Rust source.

Use `apps/gateway-server` when you need authentication, readiness during model
loading, metrics, graceful draining, or container deployment.
