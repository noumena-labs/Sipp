# CogentLM Gateway Server

`apps/gateway-server` is the first-party CogentLM HTTP gateway application. It
adds TOML configuration, bearer-token policy, local/provider targets,
management routes, metrics, and the Admin Dashboard on top of the gateway
toolkit crates.

## Start Here

- Source and generated-exe workflows:
  [Gateway Server](../../docs/gateway/server.md)

  ```bash
  cp apps/gateway-server/config/local.toml.example apps/gateway-server/config/local.toml
  clm run gateway-server check --config apps/gateway-server/config/local.toml
  ```

- Docker workflows:
  [Gateway Docker](../../docs/gateway/docker.md)

  ```bash
  cp apps/gateway-server/.env.example apps/gateway-server/.env
  cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
  cp apps/gateway-server/config/development.toml.example apps/gateway-server/config/development.toml
  docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
  ```

- TOML schema and route behavior:
  [Gateway Configuration](../../docs/gateway/configuration.md)

- Raw HTTP testing:
  [Gateway Testing](../../docs/gateway/testing.md)

## Local Files

- `config/local.toml.example`: source/local host-run template.
- `config/development.toml.example`: local Docker/development-server template.
- `config/production.toml.example`: production Docker template.
- `config/provider-only.toml.example`: provider-router template with no local
  model target.
- `config/hybrid.toml.example`: local GPU model plus provider target template.
- `.env.example`: copyable secrets-only env template.
- `admin-ui/`: React Admin Dashboard built by `clm build gateway-server` and
  copied beside the generated gateway binary.
- `development.yml.example`: copyable local Compose template.
- `development-provider-only.yml.example`: local Compose template without a
  model mount.
- `production.yml.example`: production Compose template for a prebuilt image.
- `production-provider-only.yml.example`: production Compose template without a
  model mount.
- `Dockerfile`: image build for provider-router CPU images and GPU
  model-serving variants.

Private TOML, `.env`, and copied Compose files should stay out of source
control.

Dashboard observability history, rate-limit buckets, manual blocklists,
sessions, CSRF tokens, and runtime control overrides are in-memory only and
reset when the server restarts.
