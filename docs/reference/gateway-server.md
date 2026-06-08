# Gateway Server Reference

`apps/gateway-server` is configured by one TOML file. The same schema is used
for source/exe runs and Docker runs; only path and bind interpretation changes.
Use [Gateway Server](../packages/gateway-server.md) for source/exe commands and
[Gateway Server Docker](../packages/gateway-server-docker.md) for container
commands.

## Example

```toml
public_bind = "0.0.0.0:8080"
management_bind = "0.0.0.0:9090"
max_request_bytes = 1048576
max_concurrent_requests = 4
allowed_origins = []
admin_password = "replace-me"

[routes]
query = "/v1/query"
chat = "/v1/chat"
embed = "/v1/embed"
index = "/"
health = "/healthz"
readiness = "/readyz"
metrics = "/metrics"
admin = "/admin"

[[tokens]]
env = "COGENTLM_GATEWAY_TOKEN"
caller = "production-client"
targets = ["local"]

[[targets]]
name = "local"
type = "local"
model = "/models/model.gguf"
backend = "auto"
stats = "basic"
```

## Top-Level Fields

| Field | Meaning |
| --- | --- |
| `public_bind` | Address for public inference routes. Source/exe binds this on the host; Docker binds inside the container. |
| `management_bind` | Address for health, readiness, metrics, index, and admin routes. Must differ from `public_bind`. |
| `max_request_bytes` | Maximum HTTP request body size. Must be greater than zero. |
| `max_concurrent_requests` | Optional application-wide request admission limit. Omit for unbounded. |
| `allowed_origins` | CORS allowlist for browser requests to the public listener. Empty disables the CORS layer. |
| `admin_password` | Literal Admin Dashboard password. Required and non-blank. Keep production TOML private. |

`check` validates these fields without reading token env vars, loading models,
or binding ports.

## Routes

`query`, `chat`, and `embed` are required public routes. The other routes are
management routes:

- `index`: optional management index JSON route.
- `health`: optional liveness route returning `ok`.
- `readiness`: optional readiness route returning `ready`.
- `metrics`: optional Prometheus text route.
- `admin`: optional Admin Dashboard route. Login and logout are derived as
  `<admin>/login` and `<admin>/logout`.

Routes must be absolute paths and must not contain query strings or fragments.
Public routes cannot duplicate each other. Management routes, including derived
admin login/logout routes, cannot duplicate each other.

## Tokens

Each `[[tokens]]` block maps one bearer-token environment variable to a caller
label and a target allowlist:

```toml
[[tokens]]
env = "COGENTLM_GATEWAY_TOKEN"
caller = "browser-client"
targets = ["local", "openai-chat"]
```

- `env` names the environment variable containing the bearer token value.
- `caller` is a stable label used in request metadata and diagnostics.
- `targets` lists allowed `[[targets]].name` values. An empty list grants all
  configured targets.

Token values must be non-empty and contain no whitespace. They are read only
when `serve` starts.

## Targets

Each `[[targets]]` block publishes one model or provider endpoint under a
stable target name.

### Local GGUF

```toml
[[targets]]
name = "local"
type = "local"
model = ".build/models/qwen2.5-0.5b-instruct-q4_0.gguf"
backend = "auto"
stats = "basic"
```

- `model` is the GGUF path seen by the process. Relative paths resolve from
  the process working directory.
- `backend` can be `auto`, `cpu`, `cuda`, `metal`, or `vulkan`.
- `stats` can be `off`, `basic`, or `profile`.
- `runtime` can contain advanced native runtime settings from the shared
  runtime options schema.

`backend = "auto"` selects the best compiled and available backend in this
order: CUDA, Metal, Vulkan, then CPU. Explicit `cpu` disables GPU offload.
Explicit GPU backends fail if that backend was not compiled or is unavailable.

`stats = "off"` disables runtime metrics and backend profiling.
`stats = "basic"` enables runtime metrics. `stats = "profile"` enables runtime
metrics and backend profiling.

### OpenAI

```toml
[[targets]]
name = "openai-chat"
type = "openai"
model = "provider-model"
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"
timeout_seconds = 60
```

`base_url` and `timeout_seconds` are optional. The API key is read from
`api_key_env` when `serve` starts.

### OpenAI-Compatible

```toml
[[targets]]
name = "compatible-chat"
type = "openai_compatible"
model = "served-model"
base_url = "https://provider.example/v1"
token_env = "PROVIDER_TOKEN"
correlation_header = "x-request-id"
timeout_seconds = 60
```

`base_url` and `token_env` are required. `correlation_header` and
`timeout_seconds` are optional.

### Anthropic

```toml
[[targets]]
name = "anthropic-chat"
type = "anthropic"
model = "provider-model"
api_key_env = "ANTHROPIC_API_KEY"
version = "2023-06-01"
timeout_seconds = 60
```

`base_url`, `version`, and `timeout_seconds` are optional. The API key is read
from `api_key_env` when `serve` starts.

## Bind Behavior

Source/exe mode binds `public_bind` and `management_bind` directly on the host.
Docker mode binds those addresses inside the container; Compose `ports` decide
host exposure.

For Docker:

- Local testing keeps both host ports on `127.0.0.1`.
- Production exposes public traffic through the configured host port and keeps
  management on `127.0.0.1` by default.
- Local model paths should match the container mount point:
  `/workspace/.build/models` for development Compose and `/models` for
  production Compose.

## Admin Dashboard

The dashboard is served only on the management listener. It uses
`admin_password` for login, stores short-lived HTTP-only sessions, and does not
render the password, bearer tokens, or provider secrets.
