# CogentLM Gateway Server

`cogentlm-gateway-server` is the production, headless middle layer for hosting
local or provider-backed CogentLM aliases.

Public listener:

- `POST /v1/query`
- `POST /v1/chat`
- `POST /v1/embed`

Management listener:

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

The management listener binds before endpoint loading. `/healthz` remains 200
while the process is responsive; `/readyz` remains 503 until every configured
alias is loaded, and becomes 503 again during draining.

## Run

```bash
export COGENTLM_GATEWAY_TOKEN='replace-me'
cargo run -p cogentlm-gateway-server -- \
  check --config apps/gateway-server/config/production.toml
cargo run -p cogentlm-gateway-server -- \
  serve --config apps/gateway-server/config/production.toml
```

`check` only parses and validates configuration. It does not read secrets,
load models, or contact providers.

## Shutdown

The defaults allow 120 seconds for active inference to finish, then cancel
remaining work and allow 5 seconds for terminal stream errors to flush.
Uncommitted unary requests receive HTTP 503. Committed SSE responses receive a
terminal `error` event with code `server_restarting`.

Client disconnects cancel the associated native or provider execution
immediately.

## Deployment

Build the CPU image from the repository root:

```bash
docker build -f apps/gateway-server/Dockerfile -t cogentlm-gateway:cpu .
docker compose -f apps/gateway-server/compose.yaml up
```

The image runs as a non-root user. Mount models read-only and expose the
management listener only to the monitoring network. See
`deploy/nginx.conf` for reverse-proxy timeout and buffering guidance.
