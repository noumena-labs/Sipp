# Gateway Docker

This page covers local Docker testing and production Docker deployment for the
CogentLM Gateway Server. Source and generated-executable workflows are
documented in [Gateway Server](server.md).

Docker workflows use raw `docker` commands. The image runs:

```bash
cogentlm-gateway serve --config /etc/cogentlm/gateway.toml
```

The Dockerfile invokes the workspace `xtask` package directly with
`cargo run --package xtask -- ...`; it does not rely on the local
`cargo xtask` alias from `.cargo/config.toml`.

## Files

- `apps/gateway-server/Dockerfile` builds the staged gateway distribution.
- `apps/gateway-server/development.yml.example` is the local Compose template.
- `apps/gateway-server/development-provider-only.yml.example` is the local
  provider-router Compose template with no model mount.
- `apps/gateway-server/production.yml.example` is the production Compose
  template for a prebuilt local or registry image.
- `apps/gateway-server/production-provider-only.yml.example` is the production
  provider-router Compose template with no model mount.
- `apps/gateway-server/.env.example` is a copyable Compose env starting point.
- `apps/gateway-server/config/local.toml.example` is for source/local host
  runs.
- `apps/gateway-server/config/development.toml.example` is for local Docker
  and development-server runs.
- `apps/gateway-server/config/production.toml.example` is for production
  Docker runs.
- `apps/gateway-server/config/provider-only.toml.example` is for gateways that
  route only to provider APIs.
- `apps/gateway-server/config/hybrid.toml.example` is for gateways that serve
  a local GPU model and route to provider APIs.

## On-Board GPU Docker Testing

Use the model-serving Docker template when validating on-board GPU inference
and Compose wiring on a workstation.

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
cp apps/gateway-server/config/development.toml.example apps/gateway-server/config/development.toml
```

Edit `apps/gateway-server/config/development.toml`:

- Set `admin_password` to the local Admin Dashboard password.
- Set `model` to the path the container will see. The development Compose
  mount exposes the host model directory as `/models`.
- Keep `public_bind = "0.0.0.0:8080"` and
  `management_bind = "0.0.0.0:9090"` so the process listens on the container
  network interface.

Edit `apps/gateway-server/.env`:

- Set `COGENTLM_GATEWAY_CONFIG=./config/development.toml`.
- Keep `COGENTLM_GATEWAY_RUNTIME_ENV_FILE=./.env` so the copied env file is
  also injected into the gateway container.
- Set `COGENTLM_MODEL_DIR` to the host directory containing the configured
  `.gguf` file.
- Set `COGENTLM_GATEWAY_TOKEN` to the bearer token used by test clients.
- Add any provider secret env vars named by TOML targets, for example
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or a compatible provider token.

Build and run:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=vulkan \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:vulkan .
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up
```

For other GPU backends, set `COGENTLM_GATEWAY_BACKEND` and
`COGENTLM_GATEWAY_IMAGE` to the backend you want to build, such as `cuda` or
`metal`.

The development Compose file maps both host ports to `127.0.0.1`, so the
gateway stays local to the workstation even though the process binds
`0.0.0.0` inside the container.

Open `http://127.0.0.1:9090/admin` and log in with the TOML
`admin_password`. Send client requests to `http://127.0.0.1:8080`.

## Provider-Only Docker Testing

Use the provider-only template when the gateway should hold provider secrets
and route prompts upstream without loading a local GGUF model:

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development-provider-only.yml.example apps/gateway-server/development-provider-only.yml
cp apps/gateway-server/config/provider-only.toml.example apps/gateway-server/config/provider-only.toml
```

Edit `apps/gateway-server/config/provider-only.toml`:

- Set `admin_password` to the local Admin Dashboard password.
- Keep `public_bind = "0.0.0.0:8080"` and
  `management_bind = "0.0.0.0:9090"` so the process listens on the container
  network interface.
- Do not add a local target unless this gateway is becoming hybrid.

Edit `apps/gateway-server/.env`:

- Set `COGENTLM_GATEWAY_CONFIG=./config/provider-only.toml`.
- Keep `COGENTLM_GATEWAY_RUNTIME_ENV_FILE=./.env`.
- Set `COGENTLM_GATEWAY_TOKEN` to the bearer token used by test clients.
- Set `OPENAI_API_KEY` for the checked-in provider-only TOML, or replace the
  target with another supported provider and set that provider's secret env
  var.
- Leave `COGENTLM_MODEL_DIR` unused.

Build and run the provider-router image:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=cpu \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:provider-cpu .
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml config
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml up
```

CPU is appropriate here because no local inference target is loaded.

## Production Docker Deployment

Use production Docker deployment when running a prebuilt local image or a
private-registry image. The production Compose file does not build from
source.

Prepare private production files from the examples:

```bash
cp apps/gateway-server/production.yml.example /opt/cogentlm/gateway/production.yml
cp apps/gateway-server/config/production.toml.example /opt/cogentlm/gateway/production.toml
```

Set a real `admin_password` in `/opt/cogentlm/gateway/production.toml` and
keep that file outside the repo.

Prepare a private env file from `apps/gateway-server/.env.example`, then set:

- `COGENTLM_GATEWAY_IMAGE`: the prebuilt local or registry image.
- `COGENTLM_GATEWAY_CONFIG`: host path to the private production TOML file.
- `COGENTLM_GATEWAY_RUNTIME_ENV_FILE`: absolute path to this private env file,
  so Compose passes bearer and provider secrets into the container.
- `COGENTLM_MODEL_DIR`: host directory mounted at `/models` for local or
  hybrid model-serving configs.
- `COGENTLM_GATEWAY_TOKEN`: production bearer token value.
- Provider secret variables referenced by production TOML targets.
- `COGENTLM_GATEWAY_PUBLIC_PORT` and `COGENTLM_GATEWAY_MANAGEMENT_PORT` as
  needed.

Build or publish the image:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=vulkan \
  -f apps/gateway-server/Dockerfile \
  -t registry.example.com/cogentlm-gateway:vulkan .
```

For other backends, change the `COGENTLM_GATEWAY_BACKEND` build argument and
`COGENTLM_GATEWAY_IMAGE` tag accordingly.

Deploy:

```bash
docker compose --env-file /opt/cogentlm/gateway/gateway.env -f /opt/cogentlm/gateway/production.yml config
docker compose --env-file /opt/cogentlm/gateway/gateway.env -f /opt/cogentlm/gateway/production.yml up -d
```

The production template publishes public traffic on the configured host port
and binds the management port to `127.0.0.1` on the host by default.

For provider-only production, copy
`apps/gateway-server/production-provider-only.yml.example` instead of
`production.yml.example`, copy `config/provider-only.toml.example`, set
`COGENTLM_GATEWAY_CONFIG` to that private TOML path, and omit
`COGENTLM_MODEL_DIR`.

## Bind And Mount Behavior

The TOML file always uses the same schema, but bind and path interpretation
changes by runtime mode.

| Runtime | TOML bind values | Host exposure | Local target `model` path |
| --- | --- | --- | --- |
| Source/exe | Host addresses, usually `127.0.0.1:*` for development | The process binds directly on the host | Path seen from the process working directory |
| Local Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | `development.yml` maps host ports to `127.0.0.1` | `/models/<file>.gguf` |
| Production Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | `production.yml` exposes public and keeps management host-local by default | `/models/<file>.gguf` |
| Provider-only Compose | Container addresses, usually `0.0.0.0:8080` and `0.0.0.0:9090` | Provider-only templates follow the same port rules | No local model path |

Compose mount variables:

| Variable | Host value | Container path |
| --- | --- | --- |
| `COGENTLM_GATEWAY_CONFIG` | TOML file path | `/etc/cogentlm/gateway.toml` |
| `COGENTLM_MODEL_DIR` | Directory containing local GGUF files | `/models` |

`COGENTLM_MODEL_DIR` is required only for local or hybrid model-serving
configs. Provider-only Compose templates do not mount `/models`.

Compose env behavior:

- `docker compose --env-file <path>` supplies variables used to render the
  Compose file, such as image names, ports, and mount paths.
- `COGENTLM_GATEWAY_RUNTIME_ENV_FILE` points at the env file that is injected
  into the container. Put bearer tokens and provider secrets there.
- Explicit `environment` values in the Compose file keep
  `COGENTLM_GATEWAY_TOKEN` and `RUST_LOG` visible in rendered
  `docker compose config` output; provider secrets are passed through the
  runtime env file.

Keep management private in production. Put public ingress, TLS, and external
auth controls in front of the public listener when needed.

## GPU Image Builds

Provider-only router:

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=cpu \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:provider-cpu .
```

Use CPU images for provider-only routing, not for production on-board
inference.

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
