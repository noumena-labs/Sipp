# Python Package

The Python package target is `cogentlm`. It exposes native descriptor classes,
run handles, token streaming, and the same endpoint model as the Rust client.

## Use It For

- Python applications that need local GGUF inference.
- Gateway-backed inference from Python services or scripts.
- Direct provider descriptors where server-side credentials are appropriate.
- Source-based package validation through maturin and xtask.

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

## Gateway

Register `GatewayDescriptor` when a Python service or script calls a separate
CogentLM gateway. Gateway examples live under `examples/python`.

## Related Docs

- [Installation](../getting-started/installation.md)
- [Local Inference](../guides/local-inference.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
