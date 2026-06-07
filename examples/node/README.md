# Node Examples

Each `.mjs` file demonstrates one Node.js workflow with client creation,
endpoint registration, request construction, streaming, and cleanup visible in
the file. `_support.mjs` only parses inputs and prints results.

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

Use the one-command gateway workflow when possible:

```bash
cargo xtask run examples gateway node --case query
```

For a manually started gateway, set `COGENTLM_GATEWAY_URL` and
`COGENTLM_GATEWAY_TOKEN`, then run:

```bash
node examples/node/gateway_query.mjs <model.gguf> local [input]
node examples/node/gateway_chat.mjs <model.gguf> local [input]
node examples/node/gateway_embed.mjs <model.gguf> local [input]
```

`gateway_embed` requires a model/runtime that reports embedding support.

See [../README.md](../README.md) for shared gateway and provider setup details.
