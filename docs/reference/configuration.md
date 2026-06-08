# Configuration

CogentLM configuration is intentionally split by responsibility. Core crates do
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
- `admin_password` sets the literal Admin Dashboard password in TOML.
- `[[tokens]]` maps bearer-token environment variables to caller labels and
  allowed targets.
- `[[targets]]` defines local, OpenAI, OpenAI-compatible, or Anthropic targets.
  Local targets can select `backend = "auto"`, `cpu`, `cuda`, `metal`, or
  `vulkan`. See [Gateway Configuration](../gateway/configuration.md) for the full
  schema.

Custom wire formats, authentication schemes, and route layouts belong in
separate applications composed from `lib/gateway`.

## Environment Variables

- `COGENTLM_GATEWAY_TOKEN`: development bearer token for examples and gateway
  server commands.
- `COGENTLM_MODEL_DIR`: Docker-mounted GGUF model directory for
  `apps/gateway-server`; the development template defaults to `.build/models`.
- `COGENTLM_GATEWAY_CONFIG`: host TOML path mounted by
  `apps/gateway-server/development.yml.example` and
  `apps/gateway-server/production.yml`.
- `COGENTLM_GATEWAY_IMAGE`: local or private-registry image used by the gateway
  Docker templates.
- `COGENTLM_GATEWAY_URL`: gateway base URL for client examples.
- `COGENTLM_NODE_BACKEND`: Node runtime backend selection.
- `COGENTLM_PYTHON_BACKEND`: Python runtime backend selection.
- `OPENAI_API_KEY`: provider credential used by OpenAI examples and
  provider-backed gateway targets.
