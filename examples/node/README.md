# Node Examples

Each `.mjs` file demonstrates one workflow with the `CogentClient` construction,
endpoint registration, request construction, and streaming visible in the file.
`_support.mjs` only parses inputs and prints results.

Endpoints use the unified descriptor API:

```js
const endpoint = await client.add('local', {
  kind: 'local',
  modelPath,
  config: runtime,
});
```

## Local GGUF

Build the Node binding if needed:

```bash
cargo xtask build node --backend cpu
```

Run:

```bash
node examples/node/query.mjs <model.gguf> [input]
node examples/node/chat.mjs <model.gguf> [input]
node examples/node/embed.mjs <model.gguf> [input]
node examples/node/vision_chat.mjs <model.gguf> <projector.gguf> <image> [input]
```

Set `COGENTLM_NODE_BACKEND=cpu|vulkan|cuda|metal` to choose a built backend.

## Gateway Clients

To start the local gateway and run one Node gateway client from a single
terminal:

```bash
cargo xtask run examples gateway node --case query
```

The cached sample model under `.build/models` is used by default; pass
`--model <model.gguf>` to override it.

Start a gateway first, then set:

```bash
export COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
export COGENTLM_GATEWAY_TOKEN="dev-token"
```

Run:

```bash
node examples/node/gateway_query.mjs <model.gguf> local [input]
node examples/node/gateway_chat.mjs <model.gguf> local [input]
node examples/node/gateway_embed.mjs <model.gguf> local [input]
```

`gateway_embed` requires a model/runtime that reports embedding support.

For the OpenAI gateway, use alias `openai-chat` for query/chat and
`openai-embed` for embeddings. The OpenAI gateway requires `OPENAI_API_KEY` in
the gateway process.
