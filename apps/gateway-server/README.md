# CogentLM Gateway Server

`apps/gateway-server` is the first-party CogentLM HTTP gateway application. It
adds TOML configuration, bearer-token policy, local/provider targets,
management routes, metrics, and the Admin Dashboard on top of the gateway
toolkit crates.

## Start Here

- Source and generated-exe workflows:
  [Gateway Server](../../docs/gateway/server.md)

  ```bash
  clm run gateway-server check --config apps/gateway-server/config/development.toml
  ```

- Docker workflows:
  [Gateway Docker](../../docs/gateway/docker.md)

  ```bash
  docker compose --env-file apps/gateway-server/.env.example -f apps/gateway-server/development.yml.example config
  ```

- TOML schema and route behavior:
  [Gateway Configuration](../../docs/gateway/configuration.md)

- Raw HTTP testing:
  [Gateway Testing](../../docs/gateway/testing.md)

## Local Files

- `config/development.toml`: source development example; copy it before local
  Docker use and adjust container bind/model paths.
- `config/production.toml`: production-oriented example; copy it before adding
  real secrets.
- `development.yml.example`: copyable local Compose template.
- `production.yml`: production Compose template for a prebuilt image.
- `Dockerfile`: image build for CPU, Vulkan, and CUDA gateway variants.
