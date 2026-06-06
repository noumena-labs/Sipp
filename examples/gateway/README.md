# Gateway Examples

The gateway is the server-side boundary for remote inference. Apps call
`/v1/query`, `/v1/chat`, and `/v1/embed` with a public alias and bearer token.
Provider keys, upstream URLs, and local GGUF model paths stay in a server
process.

## Source Map

- `crates/gateway`: reusable adapter, TOML config loader, alias policy, limits,
  and backend contracts.
- `examples/gateway/src/main.rs`: minimal application server that embeds the
  gateway HTTP router next to an application route.
- `apps/gateway-server`: ready-to-run Axum server with auth, CORS, health
  probes, dashboard, bounded in-memory history, and protocol routes.
- `crates/providers`: OpenAI, Anthropic, and OpenAI-compatible provider
  transports for server-side use.

## Run The Barebones Gateway Proxy

Use two terminals. The first terminal runs the gateway. The second terminal runs
a client example that calls both a local GGUF endpoint and the gateway alias.

For local GGUF inference, set a gateway token and let xtask generate a temporary
config with your model path:

```powershell
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

For an OpenAI-backed gateway, keep `OPENAI_API_KEY` in the gateway process:

```powershell
$env:OPENAI_API_KEY="<openai-api-key>"
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-openai --bind 127.0.0.1:8787
```

The xtask serve commands use `COGENTLM_GATEWAY_TOKEN` for both client auth and
dashboard admin auth. When running a copied config directly, set both secrets:

```powershell
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
$env:COGENTLM_GATEWAY_ADMIN_TOKEN="admin-token"
cargo run -p cogentlm-gateway-example -- --config <gateway.toml>
```

`examples/gateway/local-gateway.toml` and
`examples/gateway/openai-gateway.toml` are reference configs. Copy one and
replace `model_path` or provider settings before running it directly.

## Observe And Verify

Open `http://127.0.0.1:8787/` in a browser. The page shows gateway status,
configured aliases, recent request history, and copyable manual verification
commands. Enter the admin token to load protected status and history data.

Manual probes:

```powershell
curl.exe http://127.0.0.1:8787/healthz
curl.exe http://127.0.0.1:8787/readyz
curl.exe -H "Authorization: Bearer admin-token" http://127.0.0.1:8787/admin/api/status
```

Manual query:

```powershell
curl.exe -X POST http://127.0.0.1:8787/v1/query `
  -H "Authorization: Bearer dev-token" `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"local\",\"prompt\":\"Write one sentence about CogentLM.\"}"
```

## Embed The Gateway In A Server

`crates/gateway` exposes the adapter and config building blocks. Existing Axum
servers can build a `GatewayHttpService` from `apps/gateway-server`, merge the
router, and keep their own middleware, deployment, TLS, and application routes.

```powershell
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
$env:COGENTLM_GATEWAY_ADMIN_TOKEN="admin-token"
cargo run -p cogentlm-gateway-example -- --config <gateway.toml>
```

The embedded example responds to `/app-healthz` and serves the gateway protocol
and dashboard from the same listener.

## Run Clients

In a second terminal:

```powershell
$env:COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
$env:COGENTLM_GATEWAY_TOKEN="dev-token"
cargo run -p cogentlm-rust-examples --features remote --bin gateway_query -- <model.gguf> local
node examples/node/gateway_chat.mjs <model.gguf> local
python examples/python/gateway_embed.py <model.gguf> local-embed
```

Use alias `local` for query/chat examples and `local-embed` for embedding
examples. Embedding examples require a model/runtime that reports embedding
support.

OpenAI gateway aliases are `openai-chat` for query/chat and `openai-embed` for
embeddings. OpenAI workflows are manual because they require a secret and spend
provider quota.
