# Gateway Docker

Gateway Docker workflows use explicit Compose files plus the gateway TOML and a
secrets-only `.env` file.

The separation is strict:

- `.env` contains secret values only.
- TOML contains gateway application configuration.
- Compose YAML contains Docker build, image, port, mount, healthcheck, and
  container orchestration settings.

The container runs:

```bash
cogentlm-gateway serve --config /etc/cogentlm/gateway.toml
```

## Files

- `apps/gateway-server/Dockerfile` builds the staged gateway distribution.
- `apps/gateway-server/.env.example` is the secrets-only env template.
- `apps/gateway-server/development.yml.example` builds and runs a local
  model-serving image.
- `apps/gateway-server/development-provider-only.yml.example` builds and runs a
  provider-router image with no model mount.
- `apps/gateway-server/production.yml.example` runs a prebuilt production
  model-serving image.
- `apps/gateway-server/production-provider-only.yml.example` runs a prebuilt
  provider-router image with no model mount.
- `apps/gateway-server/config/*.toml.example` are gateway application config
  templates.

## Local Model-Serving Docker

From the repository root:

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
cp apps/gateway-server/config/development.toml.example apps/gateway-server/config/development.toml
```

Edit `apps/gateway-server/.env` and set only secrets:

```bash
COGENTLM_GATEWAY_ADMIN_PASSWORD=replace-me
COGENTLM_GATEWAY_TOKEN=replace-me
OPENAI_API_KEY=replace-me
ANTHROPIC_API_KEY=replace-me
```

Edit `apps/gateway-server/config/development.toml`:

- Set the local target `model` to the path the container sees, usually
  `/models/<file>.gguf`.
- Keep `public_bind = "0.0.0.0:8080"` and
  `management_bind = "0.0.0.0:9090"` so the gateway listens inside the
  container.
- Keep `admin_password_env = "COGENTLM_GATEWAY_ADMIN_PASSWORD"` unless the
  `.env` secret name also changes.

Edit `apps/gateway-server/development.yml` for Docker concerns such as image
tag, build backend, build images, model mount, port publishing, and
healthcheck.

Build and run:

```bash
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up
```

## Provider-Only Docker

Provider-only Docker runs use the provider-only Compose template and no model
mount:

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development-provider-only.yml.example apps/gateway-server/development-provider-only.yml
cp apps/gateway-server/config/provider-only.toml.example apps/gateway-server/config/provider-only.toml
```

Set secrets in `apps/gateway-server/.env`, then run:

```bash
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml config
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml build
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml up
```

The provider-only template builds a CPU gateway image because inference happens
upstream.

## Production Docker

Keep production TOML, Compose, and `.env` copies outside the repository:

```bash
mkdir -p /opt/cogentlm/gateway
cp apps/gateway-server/.env.example /opt/cogentlm/gateway/.env
cp apps/gateway-server/production.yml.example /opt/cogentlm/gateway/production.yml
cp apps/gateway-server/config/production.toml.example /opt/cogentlm/gateway/production.toml
```

Edit `/opt/cogentlm/gateway/.env` for secret values only. Edit
`/opt/cogentlm/gateway/production.toml` for gateway runtime configuration.
Edit `/opt/cogentlm/gateway/production.yml` for image names, host model
mounts, ports, restart policy, and healthcheck.

Deploy:

```bash
docker compose --env-file /opt/cogentlm/gateway/.env -f /opt/cogentlm/gateway/production.yml config
docker compose --env-file /opt/cogentlm/gateway/.env -f /opt/cogentlm/gateway/production.yml up -d
```

For provider-only production, copy `production-provider-only.yml.example` and
`config/provider-only.toml.example` instead.

## Bind And Mount Behavior

The TOML file always uses the same schema, but bind and path interpretation
changes by runtime mode.

| Runtime | TOML bind values | Host exposure | Local target `model` path |
| --- | --- | --- | --- |
| Source/exe | Host addresses, usually `127.0.0.1:*` for development | The process binds directly on the host | Path seen from the process working directory |
| Local Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | Compose `ports` map host ports to `127.0.0.1` in local templates | `/models/<file>.gguf` |
| Production Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | Compose exposes public and keeps management host-local by default | `/models/<file>.gguf` |
| Provider-only Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | Provider-only templates follow the same port rules | No local model path |

Keep management private in production. Put public ingress, TLS, and external
auth controls in front of the public listener when needed.

## Raw Docker Build

Raw Docker commands are supported as an escape hatch. Supply every build arg
explicitly:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=vulkan \
  --build-arg COGENTLM_GATEWAY_BUILDER_IMAGE=rust:bookworm \
  --build-arg COGENTLM_GATEWAY_RUNTIME_IMAGE=debian:bookworm-slim \
  --build-arg COGENTLM_GATEWAY_INSTALL_RUSTUP=0 \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:vulkan .
```

CUDA requires NVIDIA host drivers and container runtime support. Vulkan
requires host GPU device access plus Vulkan loader and driver support. Metal is
a macOS bare-metal backend and is not available from Linux Docker.

## Health Check

The Compose templates probe the management readiness route:

```bash
curl --fail --silent http://127.0.0.1:9090/readyz
```

If you change the readiness route in TOML, update the Compose healthcheck too.
