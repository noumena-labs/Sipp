# CogentLM Gateway Server

`apps/gateway-server` is the first-party CogentLM HTTP gateway application. It
adds TOML configuration, bearer-token policy, local/provider targets,
management routes, metrics, and the Admin Dashboard on top of the gateway
toolkit crates.

## Start Here

- Source and generated-exe workflows:
  [Gateway Server](../../docs/packages/gateway-server.md)

  ```bash
  clm run gateway-server check --config apps/gateway-server/config/development.toml
  ```

- Docker workflows:
  [Gateway Server Docker](../../docs/packages/gateway-server-docker.md)

  ```bash
  docker compose --env-file apps/gateway-server/.env.example -f apps/gateway-server/development.yml.example config
  ```

- TOML schema and route behavior:
  [Gateway Server Reference](../../docs/reference/gateway-server.md)

## Local Files

- `config/development.toml`: source and local Docker example.
- `config/production.toml`: production-oriented example; copy it before adding
  real secrets.
- `development.yml.example`: copyable local Compose template.
- `production.yml`: production Compose template for a prebuilt image.
- `Dockerfile`: image build for CPU, Vulkan, and CUDA gateway variants.
