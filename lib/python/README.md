# CogentLM Python Package

## What this library is for

`lib/python` is the Python package source for `cogentlm`. It loads the best
available staged native backend and exposes the same endpoint model as the Rust
client: local GGUF models, gateway endpoints, and direct provider descriptors
registered with `CogentClient.add`.

Python callers use native descriptor classes such as `LocalModelDescriptor`,
`GatewayDescriptor`, and `ProviderDescriptor`. Text and embedding calls return
run handles; call `.result()` for the final response and use `.tokens()` for
streamed text batches.

## Getting Started

Start a CogentLM gateway on `127.0.0.1:8787`, set `COGENTLM_GATEWAY_TOKEN`, and
run:

```python
import os
from cogentlm import CogentClient, CogentTextOptions, GatewayDescriptor
client = CogentClient()
gateway = client.add("gateway", GatewayDescriptor("local", "http://127.0.0.1:8787", authentication_kind="bearer", authentication_value=os.environ["COGENTLM_GATEWAY_TOKEN"]))
print(client.query("Explain gateway inference in one sentence.", endpoint=gateway, options=CogentTextOptions(max_tokens=64)).result()["text"])
```

`GatewayDescriptor` encodes the target as the profile `model` field and sends
requests to the gateway's query, chat, and embedding routes.

## Gateway And Hybrid Inference

Hybrid inference registers a local endpoint and a gateway endpoint in the same
client. Your application chooses the endpoint per request.

```python
import os
import sys

from cogentlm import (
    CacheRuntimeConfig,
    CogentClient,
    CogentTextOptions,
    ContextRuntimeConfig,
    GatewayDescriptor,
    LocalModelDescriptor,
    LocalTextOptions,
    NativeRuntimeConfig,
    ObservabilityRuntimeConfig,
    SchedulerRuntimeConfig,
    set_llama_log_quiet,
)


def runtime_config() -> NativeRuntimeConfig:
    return NativeRuntimeConfig(
        context=ContextRuntimeConfig(n_ctx=2048),
        scheduler=SchedulerRuntimeConfig(
            continuous_batching=True,
            prefill_chunk_size=0,
        ),
        cache=CacheRuntimeConfig(mode="live_slot_prefix"),
        observability=ObservabilityRuntimeConfig(runtime_metrics=True),
    )


set_llama_log_quiet(True)
client = CogentClient()
local = client.add("local", LocalModelDescriptor(sys.argv[1], runtime_config()))
gateway = client.add(
    "gateway",
    GatewayDescriptor(
        "local",
        os.environ.get("COGENTLM_GATEWAY_URL", "http://127.0.0.1:8787"),
        authentication_kind="bearer",
        authentication_value=os.environ["COGENTLM_GATEWAY_TOKEN"],
    ),
)

prompt = sys.argv[2] if len(sys.argv) > 2 else "Compare local and gateway inference."
local_run = client.query(
    prompt,
    endpoint=local,
    options=CogentTextOptions(max_tokens=96, temperature=0.7),
    local=LocalTextOptions(context_key="python-local"),
    emit_tokens=True,
)
for batch in local_run.tokens():
    print(batch["text"], end="", flush=True)
local_response = local_run.result()

gateway_response = client.query(
    prompt,
    endpoint=gateway,
    options=CogentTextOptions(max_tokens=96, temperature=0.7),
).result()

print("\nlocal:", local_response["text"])
print("gateway:", gateway_response["text"])
```

Use `endpoint_options={...}` for gateway-specific profile extensions on an
individual request, or `protocol_options={...}` on `GatewayDescriptor` for
options attached to every gateway request. Use `provider_options={...}` only
with direct provider endpoints. Local-only options such as `LocalTextOptions`
are rejected for gateway and provider endpoints.

Set `COGENTLM_PYTHON_BACKEND=cpu|vulkan|cuda|metal` to choose a staged backend,
or leave it unset for automatic selection.
