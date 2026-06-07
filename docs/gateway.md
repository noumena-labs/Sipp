# Gateway Architecture

CogentLM has three gateway surfaces with separate responsibilities.

## `crates/gateway`

Framework-neutral protocol, alias routing, access, replica-local limits,
request context, cancellation, and execution primitives over
`cogentlm-client`. It contains no HTTP listener or deployment configuration.

## `examples/gateway`

A minimal local-model Axum example. It is intentionally unsuitable as a
production service and accepts only `--model` and `--bind`.

## `apps/gateway-server`

The production headless service. It owns public and management listeners,
scoped environment-backed bearer tokens, CORS, configuration, lifecycle,
structured logs, Prometheus metrics, container deployment, and graceful
shutdown.

During shutdown, readiness becomes false before new work is rejected. Active
requests drain for the configured deadline, then their shared gateway context
is cancelled. Local response receivers and provider HTTP tasks are dropped,
which stops inference and releases replica-local concurrency permits.

## Next.js

`@noumena-labs/cogentlm-server/next` exposes `createNextGateway` for App Router
on the Node.js runtime. It uses the same native client cancellation behavior
and maps `Request.signal` and response-body cancellation to
`client_disconnected`.
