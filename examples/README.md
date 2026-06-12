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
4. Run `provider_chat` for direct provider calls from a trusted server-side
   process.

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

## Direct Provider Examples

Direct provider examples call the selected provider from the current trusted
process without a gateway. By default they use the `gemini` preset, which maps
to CogentLM's OpenAI-compatible provider descriptor.

```bash
export COGENTLM_PROVIDER="gemini"
export GEMINI_API_KEY="<gemini-api-key>"
cargo run -p cogentlm-rust-examples --bin provider_chat -- [input]
node examples/node/provider_chat.mjs [input]
python examples/python/provider_chat.py [input]
```

For any OpenAI-compatible provider, pass the generic descriptor fields:

```bash
export COGENTLM_PROVIDER="openai_compatible"
export COGENTLM_PROVIDER_BASE_URL="https://provider.example/v1"
export COGENTLM_PROVIDER_API_KEY="<provider-api-key>"
export COGENTLM_PROVIDER_MODEL="<provider-model>"
```

Use direct providers only in trusted server-side runtimes. Browser code should
call a gateway or application route instead of holding provider credentials.

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

Direct provider examples are documented and manual because they require a real
provider API key.

See [docs/examples-demos.md](../docs/examples-demos.md) for the documentation
index.
