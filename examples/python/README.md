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
python examples/python/listen.py <asr.gguf> <projector.gguf> <audio>
python examples/python/speak.py <tts.gguf> <projector.gguf> <output.wav> [text]
```

Set `SIPP_LANGUAGE` for an ASR/TTS language hint and
`SIPP_SPEAKER_AUDIO` for optional speaker-conditioned synthesis.

Note that if you encountered `ModuleNotFoundError: No module named 'sipp'`,
use `.\lib\python\.venv\Scripts\python`.

Set `SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal` to choose a built backend.

## Gateway Clients

Use the one-command gateway workflow when possible:

```bash
cargo xtask run examples gateway python --case query
```

For a manually started gateway, set `SIPP_GATEWAY_URL` and
`SIPP_GATEWAY_TOKEN`, then run:

```bash
python examples/python/gateway_query.py <model.gguf> local [input]
python examples/python/gateway_chat.py <model.gguf> local [input]
python examples/python/gateway_embed.py <model.gguf> local [input]
```

`gateway_embed` requires a model/runtime that reports embedding support.

## Direct Provider Chat

Direct provider examples call the selected provider from the Python process
without a gateway. By default they use the `gemini` preset, which maps to
Sipp's OpenAI-compatible provider endpoint.

```bash
export SIPP_PROVIDER="gemini"
export GEMINI_API_KEY="<gemini-api-key>"
python examples/python/provider_chat.py [input]
```

For any OpenAI-compatible provider, pass the generic endpoint fields:

```bash
export SIPP_PROVIDER="openai_compatible"
export SIPP_PROVIDER_BASE_URL="https://provider.example/v1"
export SIPP_PROVIDER_API_KEY="<provider-api-key>"
export SIPP_PROVIDER_MODEL="<provider-model>"
python examples/python/provider_chat.py [input]
```

See [../README.md](../README.md) for shared gateway and provider setup details.
