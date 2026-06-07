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

Use `xtask` to start the local GGUF gateway and run a gateway client from one
terminal:

```bash
cargo xtask run examples gateway rust --case query
cargo xtask run examples gateway node --case chat
cargo xtask run examples gateway python --case embed
cargo xtask run examples gateway web
```

The command uses token `dev-token` by default, starts the gateway on
`127.0.0.1:8787`, waits for readiness, and stops the gateway after Rust, Node,
or Python clients exit. The web workflow keeps the gateway running while Vite is
serving. The cached sample model under `.build/models` is used by default; pass
`--model <model.gguf>` to override it.

The manual flow is still available when you want two terminals:

1. Start a gateway.
2. Start an app/client that calls the gateway alias.

Local GGUF gateway:

```bash
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

Open `http://127.0.0.1:8787/` to inspect the minimal example proxy page.
`examples/gateway/README.md` explains how it is built from `crates/gateway`;
production lifecycle, authentication, metrics, and deployment live in
`apps/gateway-server`.

In another terminal:

```bash
export COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo run -p cogentlm-rust-examples --features remote --bin gateway_query -- <model.gguf> local [input]
node examples/node/gateway_query.mjs <model.gguf> local [input]
python examples/python/gateway_query.py <model.gguf> local [input]
```

Use alias `local` for query, chat, and embedding. Embedding examples require a
model that reports embedding support.

Provider-backed serving belongs to the production gateway server. Start from
`apps/gateway-server/config/production.toml` and the server README. The xtask
shortcut below launches that server with generated OpenAI aliases:

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
