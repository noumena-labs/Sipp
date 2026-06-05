# Gateway Examples

The gateway is the server-side boundary for remote inference. Apps call
`/v1/query`, `/v1/chat`, and `/v1/embed` with a public alias and bearer token.
Provider keys, upstream URLs, and local GGUF model paths stay on the gateway
server.

## Source Map

- `crates/gateway/src/main.rs`: packaged `cogentlm-gateway serve --config`
  binary.
- `crates/gateway/src/config.rs`: TOML config loader and backend construction.
- `crates/gateway/src/server.rs`: Axum router, bearer auth, CORS, request
  limits, alias policy, and gateway routes.
- `crates/gateway/src/backend.rs`: gateway backend contract plus local
  CogentEngine and provider backend adapters.
- `crates/gateway-providers/src`: OpenAI, Anthropic, and OpenAI-compatible
  provider transports used behind the gateway.
- `examples/gateway/src/main.rs`: minimal application server that embeds the
  gateway router next to a normal `/healthz` endpoint.

## Run The Packaged Gateway

For local GGUF inference, set a gateway token and let xtask generate a temporary
config with your model path:

```powershell
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

The reference config is `examples/gateway/local-gateway.toml`. If you run the
packaged gateway directly, copy that file and replace `model_path` first:

```powershell
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
cargo run -p cogentlm-gateway -- serve --config <gateway.toml>
```

For an OpenAI-backed gateway, keep `OPENAI_API_KEY` in the gateway process:

```powershell
$env:OPENAI_API_KEY="<openai-api-key>"
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-openai --bind 127.0.0.1:8787
```

## Embed The Gateway In A Server

`cogentlm-gateway` exposes `GatewayFileConfig` and `GatewayService::router()`.
An existing Axum server can build the gateway service from TOML, merge the
gateway router, and keep its own routes:

```powershell
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
cargo run -p cogentlm-gateway-example -- --config <gateway.toml>
```

The example server responds to `/healthz` and serves the gateway protocol from
the same listener. This is the pattern to use when an application already owns
HTTP middleware, deployment, TLS termination, or other service routes.

## Run Clients

Gateway clients now also load a local GGUF endpoint, then call the local and
gateway endpoints through the same `CogentClient` operation:

```powershell
$env:COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
cargo run -p cogentlm-rust-examples --features remote --bin gateway_query -- <model.gguf> local
node examples/node/gateway_chat.mjs <model.gguf> local
python examples/python/gateway_embed.py <model.gguf> local
```

OpenAI gateway aliases are `openai-chat` for query/chat and `openai-embed` for
embeddings. OpenAI workflows are manual because they require a secret and spend
provider quota.
