# CogentLM Examples

These examples are real integrations, not mocks. They are meant to be opened as
tutorials: each file demonstrates one workflow clearly, with CogentLM API usage
kept in the example file instead of hidden in shared helpers.

Recommended learning order:

1. `rust`, `node`, or `python`: direct local GGUF inference with `query`,
   `chat`, and `embed`.
2. `web`: browser GGUF loading and the same local workflows in Vite.
3. `gateway`: run a gateway proxy separately, then run app examples that call
   both a local model endpoint and the gateway alias.
4. `rust/openai_provider_chat.rs`: call a provider adapter directly when you
   need to inspect the server-side provider layer.

All client examples register local models, gateways, and direct providers
through `add(key, descriptor)`. The returned `EndpointRef` is passed to
`query`, `chat`, or `embed` when explicit routing is required. Reusing a key
replaces its endpoint configuration.

## Direct Local Examples

Local examples take a GGUF model path followed by optional input:

```bash
cargo run -p cogentlm-rust-examples --bin query -- <model.gguf> [input]
node examples/node/query.mjs <model.gguf> [input]
python examples/python/query.py <model.gguf> [input]
```

Swap `query` for `chat` or `embed`. `vision_chat` also takes a projector GGUF
and image path.

## Gateway Examples

Gateway examples are intentionally two-process:

1. Start a gateway.
2. Start an app/client that calls the gateway alias.

Local GGUF gateway:

```bash
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

Open `http://127.0.0.1:8787/` to inspect status, aliases, recent request
history, and manual curl verification commands. The xtask gateway serve command
uses `dev-token` as the dashboard admin token.

In another terminal:

```bash
export COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo run -p cogentlm-rust-examples --features remote --bin gateway_query -- <model.gguf> local [input]
node examples/node/gateway_query.mjs <model.gguf> local [input]
python examples/python/gateway_query.py <model.gguf> local [input]
```

Use alias `local` for `gateway_query` and `gateway_chat`; use `local-embed`
for `gateway_embed`. Embedding examples require a model/runtime that reports
embedding support.

OpenAI gateway examples require a real `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY="<openai-api-key>"
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-openai --bind 127.0.0.1:8787
```

Use alias `openai-chat` for `gateway_query` and `gateway_chat`; use
`openai-embed` for `gateway_embed`.

## Browser Examples

```bash
cargo xtask run examples serve browser
```

Open:

- `/query.html`
- `/chat.html`
- `/embed.html`
- `/gateway_local.html`
- `/gateway_query.html`
- `/gateway_chat.html`
- `/gateway_embed.html`

The browser gateway pages collect URL, token, and alias in the page because
browser code cannot read process environment variables.

## Smoke Coverage

```bash
cargo xtask test smoke suite example-rust --case query
cargo xtask test smoke suite example-node --case embed
cargo xtask test smoke suite example-python --case chat
cargo xtask test smoke suite example-gateway --case query
cargo xtask test smoke suite example-browser --case embed
cargo xtask test smoke group examples
```

OpenAI examples are documented and manual because they require a real API key.
