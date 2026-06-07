# Configuration

CogentLM configuration is intentionally split by responsibility. Core crates do
not own HTTP routes, authentication schemes, TOML files, or deployment policy.

## Runtime Configuration

Local runtime configuration belongs to the endpoint descriptor or package-level
runtime options. Common areas include context size, scheduler behavior, cache
mode, observability, sampling, and backend selection.

## Gateway Configuration

`apps/gateway-server` owns TOML configuration for the first-party gateway
application:

- `[routes]` selects public and management paths.
- `[[tokens]]` maps bearer-token environment variables to caller labels and
  allowed targets.
- `[[targets]]` defines local, OpenAI, OpenAI-compatible, or Anthropic targets.

Custom wire formats, authentication schemes, and route layouts belong in
separate applications composed from `lib/gateway`.

## Environment Variables

- `COGENTLM_GATEWAY_TOKEN`: development bearer token for examples and gateway
  server commands.
- `COGENTLM_GATEWAY_URL`: gateway base URL for client examples.
- `COGENTLM_NODE_BACKEND`: Node runtime backend selection.
- `COGENTLM_PYTHON_BACKEND`: Python runtime backend selection.
- `OPENAI_API_KEY`: provider credential used by OpenAI examples and
  provider-backed gateway targets.
