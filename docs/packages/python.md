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
run = client.query(
    "Explain CogentLM in one sentence.",
    endpoint=endpoint,
    options=CogentTextOptions(max_tokens=64),
    local=LocalTextOptions(context_key="python-local"),
)
print(run.result()["text"])
```

Set `COGENTLM_PYTHON_BACKEND=cpu|vulkan|cuda|metal` to choose a native
backend.

## Gateway Query

```python
import os

from cogentlm import CogentClient, CogentTextOptions, GatewayDescriptor


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
run = client.query(
    "Explain gateway inference.",
    endpoint=endpoint,
    options=CogentTextOptions(max_tokens=64),
)
print(run.result()["text"])
```

Gateway clients need only the gateway URL, bearer token, and public target.
Provider credentials and local model paths stay in the gateway process.

## Related Docs

- [Gateway Server](gateway-server.md)
- [Installation](../getting-started/installation.md)
- [Local Inference](../guides/local-inference.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Maintainer source builds](../maintainers/source-builds.md)
