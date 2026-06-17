# Python Package

The Python package target is `sipppy`. It installs the import package
`sipp` and exposes native descriptor classes, run handles, token streaming,
and the same endpoint model as the Rust client.

Published wheels require Python 3.10 or newer.

See the [Library API Overview](../api) for the shared `add`, `query`,
`chat`, and `embed` contracts.

## Install

```bash
pip install sipppy
```

The default wheel includes the CPU backend. Install PyPI-published GPU
backends as extras:

```bash
pip install "sipppy[vulkan]"
pip install "sipppy[metal]"
```

The backend wheels are separate PyPI distributions. For example,
`sipppy[vulkan]` installs the main `sipppy` wheel plus the matching
`sipppy-backend-vulkan` wheel for the same release version. Python code still
imports `sipp`. CUDA backend wheels are attached to GitHub releases for the
first public release and will move to PyPI after the CUDA wheel size limit is
raised.

## Use It For

- Python applications that need local GGUF inference.
- Gateway-backed inference from Python services or scripts.
- Direct provider descriptors where server-side credentials are appropriate.
- Runtime metrics and backend selection in Python services.

## Local GGUF Query

```python
import sys

from sipp import (
    CacheRuntimeConfig,
    SippClient,
    SippTextOptions,
    ContextRuntimeConfig,
    LocalModelDescriptor,
    LocalTextOptions,
    NativeRuntimeConfig,
    ObservabilityRuntimeConfig,
    SchedulerRuntimeConfig,
)


client = SippClient()
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
        "Explain Sipp in one sentence.",
        "<|assistant|>",
    ]
)
run = client.query(
    # query: raw prompt; replace markers with the target model's template.
    query_prompt,
    endpoint=endpoint,
    options=SippTextOptions(max_tokens=64),
    local=LocalTextOptions(context_key="python-local"),
)
print(run.result()["text"])
```

Set `SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal` to choose an installed native
backend. See [Runtime Options](../reference/runtime-options.md) for local
runtime config groups and request option boundaries.

## Gateway Chat

```python
import os

from sipp import ChatMessage, SippClient, SippTextOptions, GatewayDescriptor


client = SippClient()
endpoint = client.add(
    "gateway",
    GatewayDescriptor(
        os.environ["SIPP_GATEWAY_TARGET"],
        os.environ["SIPP_GATEWAY_URL"],
        authentication_kind="bearer",
        authentication_value=os.environ["SIPP_GATEWAY_TOKEN"],
    ),
)
messages = [
    ChatMessage("system", "Answer concisely."),
    ChatMessage("user", "Explain gateway inference."),
]
run = client.chat(
    messages,
    endpoint=endpoint,
    options=SippTextOptions(max_tokens=64),
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
