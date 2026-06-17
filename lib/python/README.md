# Sipp Python Package

`lib/python` is the Python package source for the public `sipppy`
distribution. It installs the import package `sipp`, loads the best available
native backend, and exposes descriptor classes for local GGUF models, gateway
endpoints, and provider endpoints.

Text and embedding calls return run handles. Call `.result()` for the final
response and `.tokens()` for streamed text batches.

## Source Checkout

From the repository root, after `source ./setup.sh`:

```bash
sipp build python --backend cpu && python examples/python/query.py <model.gguf> "Explain Sipp."
```

`sipp` forwards to `cargo xtask`; use `cargo xtask ...` with the same arguments
if the launcher is not active.

Set `SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal` to choose a native backend.
Published wheels install the CPU-capable `sipppy` distribution by default.
PyPI-published GPU backends are optional extras, for example
`pip install "sipppy[vulkan]"` or `pip install "sipppy[metal]"`. CUDA
backend wheels are attached to GitHub releases until the PyPI file-size limit
is raised.

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
run = client.query(
    "Explain Sipp in one sentence.",
    endpoint=endpoint,
    options=SippTextOptions(max_tokens=64),
    local=LocalTextOptions(context_key="python-local"),
)
print(run.result()["text"])
```

Gateway clients use `GatewayDescriptor` when a Python service or script calls a
separate Sipp gateway.

## Learn More

- [Python package docs](../../docs/packages/python.md)
- [Local inference](../../docs/guides/local-inference.md)
- [Gateway and hybrid inference](../../docs/guides/gateway-hybrid.md)
- [Python examples](../../examples/python/README.md)
