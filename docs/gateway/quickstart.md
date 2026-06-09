# Gateway Quickstart

Use the on-board local path when the gateway should load a GGUF model, or the
provider-only path when it should route requests upstream. Read
[Server](server.md) and [Docker](docker.md) before production deployment.

## On-Board Local From Source

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/config/local.toml.example apps/gateway-server/config/local.toml
```

Edit `apps/gateway-server/config/local.toml`:

- Set the local target `model` to a GGUF file visible from the workspace root.
- Keep local source binds on `127.0.0.1`.
- Keep `admin_password_env = "COGENTLM_GATEWAY_ADMIN_PASSWORD"` unless you
  also change the `.env` secret name.

Load secrets and start:

```bash
set -a
. apps/gateway-server/.env
set +a
clm run gateway-server check --config apps/gateway-server/config/local.toml --backend vulkan
clm run gateway-server serve --config apps/gateway-server/config/local.toml --backend vulkan
```

Use `cuda` for NVIDIA hosts or `metal` for macOS hosts when those are the
intended on-board inference backends.

## Provider-Only From Source

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/config/provider-only.toml.example apps/gateway-server/config/provider-only.toml
```

Set provider secrets in `apps/gateway-server/.env`, then run:

```bash
set -a
. apps/gateway-server/.env
set +a
clm run gateway-server check --config apps/gateway-server/config/provider-only.toml --backend cpu
clm run gateway-server serve --config apps/gateway-server/config/provider-only.toml --backend cpu
```

Use the request target `openai-chat` with the checked-in provider-only example.

## Docker

Docker uses one secrets-only `.env`, one gateway TOML, and one explicit Compose
file:

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
cp apps/gateway-server/config/development.toml.example apps/gateway-server/config/development.toml
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up
```

Use `development-provider-only.yml.example` and
`config/provider-only.toml.example` for provider-only Docker.

## First HTTP Request

In a second terminal:

```bash
set -a
. apps/gateway-server/.env
set +a
export GATEWAY_URL="http://127.0.0.1:8080"
export GATEWAY_MANAGEMENT_URL="http://127.0.0.1:9090"

curl --fail --silent "$GATEWAY_MANAGEMENT_URL/readyz"
curl -sS "$GATEWAY_URL/v1/query" \
  -H "Authorization: Bearer $COGENTLM_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","prompt":"Explain gateway inference.","max_tokens":64}'
```

Use `"model":"openai-chat"` for the provider-only example.

Open `http://127.0.0.1:9090/admin` and log in with the value of
`COGENTLM_GATEWAY_ADMIN_PASSWORD`.
