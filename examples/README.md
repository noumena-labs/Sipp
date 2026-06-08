# CogentLM Examples

The examples are real integrations, not mocks. Each file keeps the CogentLM
client construction, endpoint registration, request construction, streaming,
and cleanup visible in the example itself.

Use examples when you want small, copyable code. Use demos when you want a
larger browser experience.

## Learning Order

1. Run `rust`, `node`, or `python` for local GGUF `query`, `chat`, and `embed`.
2. Run `web` for browser-local GGUF loading in Vite.
3. Run `gateway` workflows to compare local endpoints with a gateway target.
4. Inspect `rust/openai_provider_chat.rs` for direct provider calls from a
   trusted server-side process.

All client examples register endpoints through `add(key, descriptor)`. The
returned endpoint reference is passed to `query`, `chat`, or `embed` when a
request uses a specific destination.

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

The one-command gateway workflows start a local gateway, run a selected client,
and stop the gateway when the client exits:

```bash
cargo xtask run examples gateway rust --case query
cargo xtask run examples gateway node --case chat
cargo xtask run examples gateway python --case embed
cargo xtask run examples gateway web
```

These commands use token `dev-token`, bind the gateway to `127.0.0.1:8787`,
wait for readiness, and use the cached sample model under `.build/models`
unless `--model <model.gguf>` is provided. The web workflow keeps the gateway
running while Vite serves the browser pages.

Manual two-terminal gateway flows are still available:

```bash
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

In another terminal:

```bash
export COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo run -p cogentlm-rust-examples --features gateway --bin gateway_query -- <model.gguf> local [input]
node examples/node/gateway_query.mjs <model.gguf> local [input]
python examples/python/gateway_query.py <model.gguf> local [input]
```

Use target `local` for query, chat, and embedding. Embedding examples require a
model/runtime that reports embedding support.

Provider-backed serving belongs to `apps/gateway-server`. The OpenAI gateway
shortcut requires `OPENAI_API_KEY` in the gateway process and exposes targets
`openai-chat` and `openai-embed`:

```bash
export OPENAI_API_KEY="<openai-api-key>"
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-openai --bind 127.0.0.1:8787
```

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

Browser gateway pages collect URL, token, and target in the page because
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

See [docs/examples-demos.md](../docs/examples-demos.md) for the documentation
index.
