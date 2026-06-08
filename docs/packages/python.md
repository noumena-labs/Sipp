# Python Package

The Python package target is `cogentlm`. It exposes native descriptor classes,
run handles, token streaming, and the same endpoint model as the Rust client.

## Install

```bash
pip install cogentlm
```

## Use It For

- Python applications that need local GGUF inference.
- Gateway-backed inference from Python services or scripts.
- Direct provider descriptors where server-side credentials are appropriate.
- Runtime metrics and backend selection in Python services.

## Local GGUF Query

```python
import sys

from cogentlm import (
    CacheRuntimeConfig,
    CogentClient,
    CogentTextOptions,
    ContextRuntimeConfig,
    LocalModelDescriptor,
    LocalTextOptions,
    NativeRuntimeConfig,
    ObservabilityRuntimeConfig,
    SchedulerRuntimeConfig,
)


client = CogentClient()
endpoint = client.add(
    "default",
    LocalModelDescriptor(
        sys.argv[1],
        NativeRuntimeConfig(
            context=ContextRuntimeConfig(n_ctx=2048),
            scheduler=SchedulerRuntimeConfig(
                continuous_batching=True,
                prefill_chunk_size=0,
            ),
            cache=CacheRuntimeConfig(mode="live_slot_prefix"),
            observability=ObservabilityRuntimeConfig(runtime_metrics=True),
        ),
    ),
)
query_prompt = "\n".join(
    [
        "<|system|>",
        "Answer concisely.",
        "<|user|>",
        "Explain CogentLM in one sentence.",
        "<|assistant|>",
    ]
)
run = client.query(
    # query: raw prompt; replace markers with the target model's template.
    query_prompt,
    endpoint=endpoint,
    options=CogentTextOptions(max_tokens=64),
    local=LocalTextOptions(context_key="python-local"),
)
print(run.result()["text"])
```

Set `COGENTLM_PYTHON_BACKEND=cpu|vulkan|cuda|metal` to choose a native
backend. See [Runtime Options](../reference/runtime-options.md) for local
runtime config groups and request option boundaries.

Use local `query` only with an already-rendered prompt template, a
completion-style/base model, or an encoder-decoder text model. Use `chat` for
role messages and runtime chat template handling.

## Gateway Chat

```python
import os

from cogentlm import ChatMessage, CogentClient, CogentTextOptions, GatewayDescriptor


client = CogentClient()
endpoint = client.add(
    "gateway",
    GatewayDescriptor(
        os.environ["COGENTLM_GATEWAY_TARGET"],
        os.environ["COGENTLM_GATEWAY_URL"],
        authentication_kind="bearer",
        authentication_value=os.environ["COGENTLM_GATEWAY_TOKEN"],
    ),
)
messages = [
    ChatMessage("system", "Answer concisely."),
    ChatMessage("user", "Explain gateway inference."),
]
run = client.chat(
    messages,
    endpoint=endpoint,
    options=CogentTextOptions(max_tokens=64),
)
print(run.result()["text"])
```

Gateway clients need only the gateway URL, bearer token, and public target.
Provider credentials and local model paths stay in the gateway process.

## Related Docs

- [Gateway Server](../gateway/server.md)
- [Installation](../getting-started/installation.md)
- [Local Inference](../guides/local-inference.md)
- [Providers](../guides/providers.md)
- [Runtime Options](../reference/runtime-options.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Maintainer source builds](../maintainers/source-builds.md)
