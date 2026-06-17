# Python Package

The Python wheel is named `sipppy`. Python code imports the `sipp` module, which
exposes native descriptor classes, run handles, token streaming, and the same
endpoint model as the Rust client.

See the [Library API Overview](../api) for the shared `add`, `query`,
`chat`, and `embed` contracts.

## Install

> [!NOTE]
> Python wheels currently ship from the project's GitHub Releases, not PyPI.
> A full PyPI release with a complete build matrix (CPU and GPU backends across
> operating systems, architectures, and Python versions, in the style of
> PyTorch's distribution matrix) is in progress. The package name `sipppy` import are stable; only the distribution channel will change.

Download the `sipppy` wheel that matches your platform, Python version, and
backend from the [GitHub Releases](https://github.com/noumena-labs/Sipp/releases)
page, then install it with pip. The default wheel includes the CPU backend:

```bash
pip install ./sipppy-<version>-<python>-<platform>.whl
```

GPU backends ship as separate backend wheels in the same release. Install the
backend wheel that matches your hardware alongside the base `sipppy` wheel.

Once the PyPI release is available, installation will use the standard extras
syntax. The default wheel includes the CPU backend; each extra pulls the
matching GPU backend wheel for the same release version:

```bash
pip install sipppy
```

The backend wheels are separate PyPI distributions. For example,
`sipp-py[cuda]` installs the main `sipp-py` wheel plus the matching
`sipp-py-backend-cuda` wheel for the same release version. Python code still
imports `sipp`.

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
