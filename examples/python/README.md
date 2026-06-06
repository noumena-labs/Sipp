# Python Examples

Each `.py` file demonstrates one workflow with client creation, endpoint
registration, request construction, and streaming visible in the file.
`_support.py` only parses inputs and prints results.

Endpoints use the unified descriptor API:

```python
endpoint = client.add("local", LocalModelDescriptor(model_path, runtime))
```

## Local GGUF

Build/install the Python package with xtask when needed:

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

Start a gateway first, then set:

```bash
export COGENTLM_GATEWAY_URL="http://127.0.0.1:8787"
export COGENTLM_GATEWAY_TOKEN="dev-token"
```

Run:

```bash
python examples/python/gateway_query.py <model.gguf> local [input]
python examples/python/gateway_chat.py <model.gguf> local [input]
python examples/python/gateway_embed.py <model.gguf> local-embed [input]
```

`gateway_embed` requires a model/runtime that reports embedding support.

For the OpenAI gateway, use alias `openai-chat` for query/chat and
`openai-embed` for embeddings. The OpenAI gateway requires `OPENAI_API_KEY` in
the gateway process.
