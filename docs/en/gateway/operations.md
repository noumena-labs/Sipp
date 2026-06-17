# Gateway Operations

The first-party gateway has one public listener and one management listener.
Keep those operational surfaces separate in deployment.

## Public Listener

The public listener serves inference routes:

- `/v1/query`
- `/v1/chat`
- `/v1/embed`

Every public request must include a bearer token accepted by the configured
`[[tokens]]` policy. The request `model` field is the public target name. The
gateway resolves that target to a local model or provider endpoint.

Put TLS, external authentication, rate limiting, and network ingress in front
of the public listener when exposing it beyond a trusted network.

## Management Listener

The management listener can serve:

- `/`: optional index JSON route.
- `/healthz`: liveness route returning `ok`.
- `/readyz`: readiness route returning `ready`.
- `/metrics`: Prometheus text metrics route.
- `/admin`: password-protected Admin Dashboard.

Keep the management listener private. In Docker production, the Compose file
binds the management host port to `127.0.0.1` by default.

## Admin Dashboard

The Admin Dashboard uses the value of the env var named by
`admin_password_env` in TOML for login. It stores short-lived HTTP-only
sessions and does not render the password, bearer tokens, or provider secrets.

Use the dashboard to inspect configured routes, targets, selected local
backends, and current request metrics. Do not expose it directly to the public
internet.

## Metrics

The metrics route renders low-cardinality Prometheus text. Current gateway
metrics include request and error counters by operation, for example:

```text
sipp_gateway_requests_total{operation="query"} 3
sipp_gateway_errors_total{operation="chat"} 1
```

Target-level local runtime metrics depend on the target `stats` setting:

- `off`: disable runtime metrics and backend profiling.
- `basic`: enable runtime metrics.
- `profile`: enable runtime metrics and backend profiling.

## Logging

The gateway uses `tracing` JSON logs. Set `RUST_LOG` in the process
environment to control verbosity:

```bash
RUST_LOG=info
RUST_LOG=debug,sipp_gateway_server=trace
```

Do not log bearer token values, provider credentials, or production TOML
contents.

## CORS

`allowed_origins` controls browser access to the public listener. An empty
array disables the CORS layer. Add only trusted browser origins:

```toml
allowed_origins = ["https://app.example.com"]
```

Browser clients should use short-lived gateway tokens supplied at runtime, not
long-lived tokens embedded in bundles.

## Secrets

The gateway uses two types of secrets:

- `admin_password_env`: TOML field naming the dashboard password env var.
- Token/provider env vars: names are configured in TOML; values are read from
  the process environment when `serve` starts.

Keep secrets env files private and outside source control. Use deployment
secret stores where available.
