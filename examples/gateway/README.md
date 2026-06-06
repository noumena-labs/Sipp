# Minimal Gateway Proxy Example

This example shows how to build a small HTTP gateway directly from
`crates/gateway`. It is intentionally self-contained so developers can see how
the gateway adapter, bearer auth, CORS, request bodies, JSON responses, and SSE
streaming fit together in one file.

For complete end-to-end Rust, Node, Python, and Web client workflows, see
`examples/README.md`.

## Source Map

- `crates/gateway`: framework-neutral config loading, alias policy, request
  validation, backend contracts, and `GatewayAdapter`.
- `examples/gateway/src/main.rs`: minimal Axum proxy built directly on
  `GatewayAdapter`.
- `apps/gateway-server`: full-featured gateway server with admin dashboard,
  request history, production-style auth/CORS/probes, and reusable
  `GatewayHttpService`.
- `crates/providers`: OpenAI, Anthropic, and OpenAI-compatible provider
  transports used by gateway backends.

## What This Example Does

`examples/gateway/src/main.rs`:

1. Loads a `GatewayFileConfig` from TOML.
2. Reads the configured gateway bearer token from the environment.
3. Builds a framework-neutral `GatewayAdapter`.
4. Creates local Axum routes for `/v1/query`, `/v1/chat`, and `/v1/embed`.
5. Converts JSON request bodies into backend requests with `into_backend()`.
6. Authenticates each request and passes a `GatewayCaller` into the adapter.
7. Converts adapter outputs into the gateway JSON/SSE wire shape.

It does not implement admin tokens, request history, or the production
dashboard. Those belong to `apps/gateway-server`.

## Run The Proxy

For local GGUF inference, set the gateway token and let xtask generate a
temporary config with your model path:

```bash
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

For an OpenAI-backed gateway, keep `OPENAI_API_KEY` in the gateway process:

```bash
export OPENAI_API_KEY="<openai-api-key>"
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-openai --bind 127.0.0.1:8787
```

You can also run a copied config directly:

```bash
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo run -p cogentlm-gateway-example -- --config <gateway.toml>
```

`examples/gateway/local-gateway.toml` and
`examples/gateway/openai-gateway.toml` are reference configs. Copy one and
replace `model_path` or provider settings before running it directly.

The reference configs still include `auth.admin_token_env` and
`limits.history_capacity` so the same files can be adapted for
`apps/gateway-server`; this minimal example ignores those fields.

## Observe And Verify

Open `http://127.0.0.1:8787/` in a browser. The page is served by this example
binary and lists the minimal routes. It is not the production gateway dashboard.

Manual probes:

```bash
curl http://127.0.0.1:8787/healthz
curl http://127.0.0.1:8787/readyz
curl http://127.0.0.1:8787/app-healthz
```

Manual query:

```bash
curl -X POST http://127.0.0.1:8787/v1/query \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","prompt":"Write one sentence about CogentLM."}'
```

Manual streaming chat:

```bash
curl -N -X POST http://127.0.0.1:8787/v1/chat \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","stream":true,"messages":[{"role":"user","content":"Say hello."}]}'
```
