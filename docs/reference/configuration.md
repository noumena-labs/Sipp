# Configuration

Sipp configuration is intentionally split by responsibility. Core crates do
not own HTTP routes, authentication schemes, TOML files, or deployment policy.

## Runtime Configuration

Local runtime configuration belongs to the endpoint descriptor or package-level
runtime options. Common areas include context size, scheduler behavior, cache
mode, observability, sampling, and backend selection. See
[Runtime Options](runtime-options.md) for the shared option map.

## Gateway Configuration

`apps/gateway-server` owns TOML configuration for the first-party gateway
application:

- `[routes]` selects public and management paths.
- `admin_password_env` names the secret env var containing the Admin Dashboard
  password.
- `[[tokens]]` maps bearer-token environment variables to caller labels and
  allowed targets.
- `[[targets]]` defines local, OpenAI, OpenAI-compatible, or Anthropic targets.
  Local targets can select `backend = "auto"`, `cpu`, `cuda`, `metal`, or
  `vulkan`. See [Gateway Configuration](../gateway/configuration.md) for the full
  schema.

Custom wire formats, authentication schemes, and route layouts belong in
separate applications composed from `lib/gateway`.

## Environment Variables

- `SIPP_GATEWAY_TOKEN`: development bearer token for examples and gateway
  server commands.
- `SIPP_GATEWAY_ADMIN_PASSWORD`: Admin Dashboard password used by gateway
  examples.
- `SIPP_GATEWAY_URL`: gateway base URL for client examples.
- `SIPP_NODE_BACKEND`: Node runtime backend selection.
- `SIPP_PYTHON_BACKEND`: Python runtime backend selection.
- `OPENAI_API_KEY`: provider credential used by OpenAI examples and
  provider-backed gateway targets.
