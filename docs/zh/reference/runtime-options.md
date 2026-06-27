# 运行时选项

运行时配置与本地推理端点绑定。请求选项在每次调用 `query`、`chat`、`embed` 时单独传入。网关和服务商的扩展参数使用独立的选项分组，每个字段所属的层级一目了然。

## 选项层级

| 层级 | 浏览器包 | Node.js 包 | 用途 |
| --- | --- | --- | --- |
| 客户端选项 | `new SippClient(options)` | 环境和进程设置 | 管理浏览器资源、Worker、缓存、后端选择 |
| 端点加载选项 | `client.add(..., { kind: 'local', options })` | `client.add(..., { kind: 'local', config })` | 模型来源、后端偏好、加载进度、原生运行时 |
| 文本请求选项 | `client.query(prompt, options)` | `client.query({ options })` | 输出长度、采样控制、流式、取消、停止词 |
| 本地请求选项 | `contextKey`, `grammar`, media, `normalize` | `local: { contextKey, grammar, media, normalize }` | 仅限本地的 Prompt 状态、文法、图片、嵌入归一化 |
| 网关扩展 | `endpointOptions` | `endpointOptions` | 网关端点实现需要的额外字段 |
| 服务商扩展 | `providerOptions` | `providerOptions` | 直连服务商请求中才会带上的专有字段 |

Python 和 Rust 通过各语言自己的描述符和配置类/结构体提供相同的功能。

## 浏览器客户端选项

浏览器的 `SippClientOptions` 影响 WebAssembly 运行时、Worker 传输和浏览器存储，但不负责选模型。

| 选项 | 说明 |
| --- | --- |
| `executionMode` | `auto` 优先使用 Worker；`worker` 强制 Worker 传输；`main-thread` 用于调试或受限环境。 |
| `wasmThreading` | `pthread` 加载内置多线程运行时。`single-thread` 仅在显式提供自定义 `moduleUrl` 和 `wasmUrl` 资源时有效。 |
| `moduleUrl`, `wasmUrl` | 重写当前选择的运行时资源 URL。两个必须一起设。 |
| `browserCache` | 控制浏览器 GGUF 缓存的 OPFS 分片阈值和加载行为。 |
| `trustedOrigins` | 允许跨域加载运行时资源。默认只允许同源。 |
| `workerUrl` | 打包工具找不到 Worker 时手动指定入口 URL。 |

## 原生运行时配置

`NativeRuntimeConfig` 将本地运行时的设置按功能分组。

| 分组 | 常见字段 | 用途 |
| --- | --- | --- |
| `placement` | `devices`, `gpu_layers`, `split_mode`, `main_gpu`, `tensor_split`, `use_mmap`, `use_mlock`, `fit_params` | 模型放置、内存映射、GPU 驻留 |
| `context` | `n_ctx`, `n_batch`, `n_ubatch`, `n_parallel`, `n_threads`, `n_threads_batch`, `flash_attention`, `offload_kqv` | 上下文窗口、批大小、微批大小、并发数、CPU 线程、注意力机制、KV 缓存 |
| `sampling` | `samplers`, `seed`, `top_k`, `top_p`, `min_p`, `temperature`, `repeat_penalty`, `mirostat`, `logit_bias` | 本地文本生成的默认采样参数 |
| `scheduler` | `continuous_batching`, `policy`, `prefill_chunk_size`, `max_running_requests`, `max_queued_requests` | 请求调度、批处理、队列限制 |
| `cache` | `mode`, `retained_prefix_tokens`, `snapshot_interval_tokens`, `max_snapshot_entries`, `max_snapshot_bytes` | 前缀 KV 缓存复用和快照 |
| `multimodal` | `projector_path`, `use_gpu`, `image_min_tokens`, `image_max_tokens` | 视觉投影器和图片 Token 配置 |
| `residency` | `max_gpu_models_per_device`, `allow_cpu_models_while_gpu_loaded`, `require_gpu_lease` | GPU 模型驻留策略，多模型 GPU 资源分配 |
| `observability` | `runtime_metrics`, `backend_profiling` | 运行时记录延迟、吞吐量和后端诊断数据 |

运行时配置设置一次即持续生效，适用于稳定的端点行为。如果某个值因提示词、用户操作或 UI 控件不同而变化，应使用请求选项，而非在配置中写死。

## 浏览器子选项

### `executionMode`

| 模式 | 说明 |
| --- | --- |
| `auto`（默认） | 优先使用 Web Worker。Worker 加载失败或浏览器不支持时回退到主线程。 |
| `worker` | 强制使用 Web Worker。 |
| `main-thread` | 在主线程直接运行。适合调试或在受限环境（如某些浏览器扩展）中运行。 |

### `wasmThreading`

| 模式 | 说明 |
| --- | --- |
| `pthread`（默认） | 加载内置多线程 WASM 运行时。需要 `SharedArrayBuffer` 和 COOP/COEP 响应头。 |
| `single-thread` | 仅用于显式提供自定义单线程 `moduleUrl` 和 `wasmUrl` 资源的场景。 |

### `browserCache`

| 选项 | 说明 |
| --- | --- |
| `opfsThreshold` | 超过这个大小（字节）的 GGUF 文件会分片写入 OPFS。默认 64 MiB。 |
| `allowDirectLoad` | 直接从网络流式加载，跳过缓存。默认从缓存读。 |

## 请求选项

### 文本请求选项

`Options` 或 `SippTextOptions` 可同时用于 `query` 和 `chat`。

| 选项 | 类型 | 说明 |
| --- | --- | --- |
| `maxTokens` | `u32` | 生成的最大 Token 数。 |
| `temperature` | `f32` | 采样温度（0-2）。设为 `0` 等价于 greedy 采样。 |
| `topP` | `f32` | 核采样（nucleus sampling）的概率阈值。 |
| `stop` | `Vec<String>` | 停止词列表，碰到任何一个就停。 |
| `emitTokens` | `bool` | `true` 时通过异步迭代器流式返回 `TokenBatch`。 |

### 本地请求选项

`local: { ... }` 只对本地端点生效，网关和服务商端点会直接忽略。

| 选项 | 类型 | 说明 |
| --- | --- | --- |
| `contextKey` | `String` | 本地 KV 缓存的上下文键。相同键的请求复用缓存中的前缀。 |
| `grammar` | `String` | GBNF 文法字符串，约束输出格式。 |
| `jsonSchema` | `Value` | JSON Schema，约束输出为合法的 JSON。 |
| `samplers` | `Vec<Sampler>` | 覆盖运行时配置中的采样器链。 |
| `seed` | `u32` | 覆盖运行时配置中的随机种子。 |
| `minP` | `f32` | 覆盖运行时配置中的 min-p 阈值。 |
| `topK` | `u32` | 覆盖运行时配置中的 top-k 值。 |
| `media` | `Vec<MediaInput>` | 多模态输入（图片等）。浏览器包支持 `File` 和 `Blob`。 |
| `normalize` | `bool` | 嵌入返回前是否归一化。仅对 `embed` 生效。 |

### 网关扩展

网关端点实现可以通过 `endpointOptions` 接收自定义字段。

```ts
// Node.js
client.query({
  endpoint,
  prompt,
  options: { maxTokens: 64 },
  endpointOptions: { custom_field: 'value' },
});
```

`endpointOptions` 会原样传给网关端点的实现代码。Sipp 官方网关不会消费这些字段，但自定义网关应用可以自行处理。

### 服务商扩展

`providerOptions` 里的字段会合并到直连服务商的请求体中。

```ts
// Node.js
client.chat({
  endpoint,
  messages,
  options: { maxTokens: 128 },
  providerOptions: {
    reasoning_effort: 'low',
  },
});
```

服务商选项不能覆盖 Sipp 已有的强类型字段（如 `model`、`messages`、`prompt`、`temperature`、`topP`/`top_p`）。这些应该在请求选项中设。

## 相关文档

- [推理操作](inference-operations.md)
- [本地推理](../guides/local-inference.md)
- [服务商](../guides/providers.md)
- [浏览器缓存](../guides/browser-caching.md)
