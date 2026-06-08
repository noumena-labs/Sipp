# Gateway Quickstart

This page gives the shortest path to a local first-party gateway and a raw
HTTP request. Read [Server](server.md) and [Docker](docker.md) before
production deployment.

## Source Checkout

From the repository root, copy the development config to an ignored local file:

```bash
cp apps/gateway-server/config/development.toml apps/gateway-server/config/local.toml
```

Edit `apps/gateway-server/config/local.toml`:

- Set `admin_password` to a local Admin Dashboard password.
- Set the local target `model` to a GGUF file that exists from the workspace
  root, for example `.build/models/model.gguf`.
- Keep `public_bind = "127.0.0.1:8080"` and
  `management_bind = "127.0.0.1:9090"` for local source runs.

Start the gateway:

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
clm run gateway-server check --config apps/gateway-server/config/local.toml
clm run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

`clm` is the setup-installed launcher for `cargo xtask`. If it is unavailable,
use `cargo xtask` with the same arguments.

## Docker

Copy the Docker inputs:

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
cp apps/gateway-server/config/development.toml apps/gateway-server/config/local.toml
```

Edit `apps/gateway-server/config/local.toml` for the container:

```toml
public_bind = "0.0.0.0:8080"
management_bind = "0.0.0.0:9090"
admin_password = "replace-me"
```

Set the local target `model` to the path inside the container. The development
Compose file mounts `COGENTLM_MODEL_DIR` at `/workspace/.build/models`, so a
typical value is:

```toml
model = "/workspace/.build/models/model.gguf"
```

Edit `apps/gateway-server/.env`:

```bash
COGENTLM_GATEWAY_CONFIG=./config/local.toml
COGENTLM_MODEL_DIR=../../.build/models
COGENTLM_GATEWAY_TOKEN=replace-me
```

Build and run:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=cpu \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:cpu .
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up
```

Compose publishes both ports on `127.0.0.1` on the host. The TOML bind values
above bind inside the container.

## First HTTP Request

In a second terminal:

```bash
export GATEWAY_URL="http://127.0.0.1:8080"
export GATEWAY_MANAGEMENT_URL="http://127.0.0.1:9090"
export COGENTLM_GATEWAY_TOKEN="replace-me"

curl --fail --silent "$GATEWAY_MANAGEMENT_URL/readyz"
curl -sS "$GATEWAY_URL/v1/query" \
  -H "Authorization: Bearer $COGENTLM_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","prompt":"Explain gateway inference.","max_tokens":64}'
```

Open `http://127.0.0.1:9090/admin` and log in with the TOML
`admin_password`.

## Client Endpoint

Gateway clients need only the base URL, public target name, and gateway
authentication value:

```ts
import { CogentClient } from 'cogentlm';

const client = new CogentClient();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'http://127.0.0.1:8080',
  authentication: { kind: 'bearer', value: await getGatewayToken() },
});
const run = client.query('Explain gateway inference.', {
  endpoint,
  maxTokens: 64,
});
console.log((await run.response).text);
await client.close();
```

