# CogentLM Gateway Server

`apps/gateway-server` is the first-party CogentLM HTTP gateway application. It
is built from `cogentlm-gateway-core` and `cogentlm-gateway`, then adds
application-owned configuration and deployment behavior.

The server owns:

- Public and management listeners.
- Query, chat, embed, index, health, readiness, and metrics routes.
- Environment-backed bearer tokens and per-token target access.
- Local and provider-backed target construction.
- Application-wide concurrency admission.
- CORS, body limits, metrics, logging, TOML, and container policy.

## Source Checkout

From the repository root, after `source ./setup.sh`:

```bash
clm build core && cargo run -p cogentlm-gateway-server -- check --config apps/gateway-server/config/production.toml
```

`clm` forwards to `cargo xtask`; use `cargo xtask ...` with the same arguments
if the launcher is not active.

## Run The Server

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
cargo run -p cogentlm-gateway-server -- \
  check --config apps/gateway-server/config/production.toml
cargo run -p cogentlm-gateway-server -- \
  serve --config apps/gateway-server/config/production.toml
```

`check` validates TOML without reading secrets or loading endpoints. `serve`
loads endpoints before binding either listener, then applies graceful HTTP
shutdown on Ctrl-C.

## Configuration

`[routes]` selects paths. `[[tokens]]` selects bearer-token environment
variables, caller labels, and allowed targets. `[[targets]]` selects local,
OpenAI, OpenAI-compatible, or Anthropic endpoints.

Custom codecs and authentication schemes belong in a separate application
composed from `lib/gateway`.

## Deployment

```bash
docker build -f apps/gateway-server/Dockerfile -t cogentlm-gateway:cpu .
docker compose -f apps/gateway-server/compose.yaml up
```

The image runs as a non-root user. Mount model files read-only and keep the
management listener private.

See the [gateway server reference](../../docs/reference/gateway-server.md) and
[gateway layering](../../docs/gateway.md) for more detail.
