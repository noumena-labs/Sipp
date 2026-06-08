# Python Examples

Each `.py` file demonstrates one Python workflow with client creation,
endpoint registration, request construction, streaming, and cleanup visible in
the file. `_support.py` only parses inputs and prints results.

## Local GGUF

Build the Python package if needed:

```bash
cargo xtask build python --backend cpu
```

Run:

```bash
python examples/python/query.py <model.gguf> [input]
python examples/python/chat.py <model.gguf> [input]
python examples/python/embed.py <model.gguf> [input]
python examples/python/vision_chat.py <model.gguf> <projector.gguf> <image> [input]
```

Set `COGENTLM_PYTHON_BACKEND=cpu|vulkan|cuda|metal` to choose a built backend.

## Gateway Clients

Use the one-command gateway workflow when possible:

```bash
cargo xtask run examples gateway python --case query
```

For a manually started gateway, set `COGENTLM_GATEWAY_URL` and
`COGENTLM_GATEWAY_TOKEN`, then run:

```bash
python examples/python/gateway_query.py <model.gguf> local [input]
python examples/python/gateway_chat.py <model.gguf> local [input]
python examples/python/gateway_embed.py <model.gguf> local [input]
```

`gateway_embed` requires a model/runtime that reports embedding support.

See [../README.md](../README.md) for shared gateway and provider setup details.
