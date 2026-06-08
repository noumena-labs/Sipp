# Gateway Docker

This page covers local Docker testing and production Docker deployment for the
CogentLM Gateway Server. Source and generated-executable workflows are
documented in [Gateway Server](server.md).

Docker workflows use raw `docker` commands. The image runs:

```bash
cogentlm-gateway serve --config /etc/cogentlm/gateway.toml
```

## Files

- `apps/gateway-server/Dockerfile` builds the staged gateway distribution.
- `apps/gateway-server/development.yml.example` is the local Compose template.
- `apps/gateway-server/production.yml` runs a prebuilt local or registry image.
- `apps/gateway-server/.env.example` is a copyable Compose env starting point.
- `apps/gateway-server/config/*.toml` configures listeners, routes, targets,
  bearer-token env names, and the literal Admin Dashboard password.

## Local Docker Testing

Use local Docker testing when validating the image and Compose wiring on a
workstation.

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
cp apps/gateway-server/config/development.toml apps/gateway-server/config/local.toml
```

Edit `apps/gateway-server/config/local.toml`:

- Set `admin_password` to the local Admin Dashboard password.
- Set `model` to the path the container will see. The development Compose
  mount exposes the host model directory as `/workspace/.build/models`.
- Use `public_bind = "0.0.0.0:8080"` and
  `management_bind = "0.0.0.0:9090"` so the process listens on the
  container network interface.

Edit `apps/gateway-server/.env`:

- Set `COGENTLM_GATEWAY_CONFIG=./config/local.toml`.
- Set `COGENTLM_MODEL_DIR` to the host directory containing the configured
  `.gguf` file.
- Set `COGENTLM_GATEWAY_TOKEN` to the bearer token used by test clients.

Build and run:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=cpu \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:cpu .
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up
```

The development Compose file maps both host ports to `127.0.0.1`, so the
gateway stays local to the workstation even though the process binds
`0.0.0.0` inside the container.

Open `http://127.0.0.1:9090/admin` and log in with the TOML
`admin_password`. Send client requests to `http://127.0.0.1:8080`.

## Production Docker Deployment

Use production Docker deployment when running a prebuilt local image or a
private-registry image. The production Compose file does not build from
source.

Prepare a private production TOML file from
`apps/gateway-server/config/production.toml`, set a real `admin_password`, and
store it outside the repo, for example
`/opt/cogentlm/gateway/production.toml`.

Prepare a private env file from `apps/gateway-server/.env.example`, then set:

- `COGENTLM_GATEWAY_IMAGE`: the prebuilt local or registry image.
- `COGENTLM_GATEWAY_CONFIG`: host path to the private production TOML file.
- `COGENTLM_MODEL_DIR`: host directory mounted at `/models`.
- `COGENTLM_GATEWAY_TOKEN`: production bearer token value.
- `COGENTLM_GATEWAY_PUBLIC_PORT` and `COGENTLM_GATEWAY_MANAGEMENT_PORT` as
  needed.

Build or publish the image:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=cpu \
  -f apps/gateway-server/Dockerfile \
  -t registry.example.com/cogentlm-gateway:cpu .
```

Deploy:

```bash
docker compose --env-file /opt/cogentlm/gateway/gateway.env -f apps/gateway-server/production.yml config
docker compose --env-file /opt/cogentlm/gateway/gateway.env -f apps/gateway-server/production.yml up -d
```

The production template publishes public traffic on the configured host port
and binds the management port to `127.0.0.1` on the host by default.

## Bind And Mount Behavior

The TOML file always uses the same schema, but bind and path interpretation
changes by runtime mode.

| Runtime | TOML bind values | Host exposure | Local target `model` path |
| --- | --- | --- | --- |
| Source/exe | Host addresses, usually `127.0.0.1:*` for development | The process binds directly on the host | Path seen from the process working directory |
| Local Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | `development.yml` maps host ports to `127.0.0.1` | `/workspace/.build/models/<file>.gguf` |
| Production Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | `production.yml` exposes public and keeps management host-local by default | `/models/<file>.gguf` |

Compose mount variables:

| Variable | Host value | Container path |
| --- | --- | --- |
| `COGENTLM_GATEWAY_CONFIG` | TOML file path | `/etc/cogentlm/gateway.toml` |
| `COGENTLM_MODEL_DIR` in development | Directory containing local GGUF files | `/workspace/.build/models` |
| `COGENTLM_MODEL_DIR` in production | Directory containing local GGUF files | `/models` |

Keep management private in production. Put public ingress, TLS, and external
auth controls in front of the public listener when needed.

## GPU Image Builds

Vulkan:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=vulkan \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:vulkan .
```

CUDA:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=cuda \
  --build-arg COGENTLM_GATEWAY_BUILDER_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --build-arg COGENTLM_GATEWAY_RUNTIME_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  --build-arg COGENTLM_GATEWAY_INSTALL_RUSTUP=1 \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:cuda .
```

CUDA requires NVIDIA host drivers and container runtime support. Vulkan
requires host GPU device access plus Vulkan loader and driver support. Metal is
a macOS bare-metal backend and is not available from Linux Docker.

## Health Check

Both Compose files probe the management readiness route:

```bash
curl --fail --silent http://127.0.0.1:9090/readyz
```

If you change the readiness route in TOML, update the Compose healthcheck too.

