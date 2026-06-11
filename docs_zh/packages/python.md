# Python 包

Python 包发布名称为 `cogentlm`。提供原生描述符类、运行句柄和 Token 流式传输，采用与 Rust 客户端相同的端点模型。

各平台共享的 `add`、`query`、`chat`、`embed` 见[API 概述](../api)。

## 安装

```bash
pip install cogentlm
```

## 适用场景

- Python 应用中执行本地推理。
- Python 服务或脚本中调用网关。
- 服务端安全管理凭证的前提下直接调用提供商 API。
- 控制 Python 服务中的运行时指标和引擎后端选择。

## 本地推理 (Query)

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
    # query 接收原始提示词；请确保提示词匹配目标模型的格式模板。
    query_prompt,
    endpoint=endpoint,
    options=CogentTextOptions(max_tokens=64),
    local=LocalTextOptions(context_key="python-local"),
)
print(run.result()["text"])
```

设置环境变量 `COGENTLM_PYTHON_BACKEND=cpu|vulkan|cuda|metal` 来选择原生后端引擎。关于本地运行时的配置参数与请求选项说明，请参阅[运行时选项](../reference/runtime-options.md)。

## 网关推理

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

网关客户端只需提供网关 URL、Bearer 凭证和公开目标名称。提供商凭证和本地模型路径均由网关进程负责管理。

## 相关文档

- [网关服务器](../gateway/server.md)
- [安装](../getting-started/installation.md)
- [本地推理](../guides/local-inference.md)
- [提供商](../guides/providers.md)
- [运行时选项](../reference/runtime-options.md)
- [网关与混合推理](../guides/gateway-hybrid.md)
- [维护者源码构建](../maintainers/source-builds.md)
