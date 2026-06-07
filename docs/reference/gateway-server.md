# Gateway Server

`apps/gateway-server` is the first-party CogentLM HTTP gateway application. It
is built from `cogentlm-gateway-core` and `cogentlm-gateway`, then adds
application-owned policy and deployment behavior.

For user-facing setup and client examples, start with
[Gateway Server](../packages/gateway-server.md).

## Source Checkout

From the repository root, after `source ./setup.sh`:

```bash
clm build core && cargo run -p cogentlm-gateway-server -- check --config apps/gateway-server/config/production.toml
```

`clm` forwards to `cargo xtask`; use `cargo xtask ...` with the same arguments
if the launcher is not active.

## Responsibilities

- Public and management listeners.
- Query, chat, embed, index, health, readiness, and metrics routes.
- Environment-backed bearer tokens and target access lists.
- Local and provider-backed target construction.
- Concurrency admission, CORS, body limits, logging, and TOML.

## Basic Commands

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
cargo run -p cogentlm-gateway-server -- \
  check --config apps/gateway-server/config/production.toml
cargo run -p cogentlm-gateway-server -- \
  serve --config apps/gateway-server/config/production.toml
```

`check` validates TOML without reading secrets or loading endpoints. `serve`
loads endpoints before binding either listener.

## More Detail

- [Gateway Architecture](../gateway.md)
- [Configuration](configuration.md)
- [Gateway server source](https://github.com/noumena-labs/CogentLM/tree/master/apps/gateway-server)
