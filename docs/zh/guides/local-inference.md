# 本地推理

本地推理指在当前浏览器、Node.js、Python、Rust 或 CLI 进程内直接运行 GGUF 模型。应用完全掌控模型选择、运行时生命周期、资源清理和向用户暴露的请求选项。

通过 `SippClient.add` 注册本地端点，保存返回的引用，在调用 `query`、`chat` 或 `embed` 时传入该引用。

## 端点调用流程

1. 选择一个支持所需功能的 GGUF 模型。
2. 通过 `client.models.add` 添加模型路径、URL 或浏览器文件。
3. 使用返回的 `ManagedModel` 注册本地端点。
4. 在端点中配置加载时的运行时选项。
5. 在调用 `query`、`chat`、`embed` 时传入请求选项。
6. 流式接收 Token 或等待最终结果。
7. 页面、Worker、服务或脚本不再需要模型时，关闭客户端释放资源。

需要显式路由时，传入 `add` 返回的不透明端点引用。只有一个兼容的本地端点时可以省略。

## 模型来源

浏览器 `models.add` 接受 `File`、字符串或 `URL`，并将内容持久化到 OPFS。Node.js、Python 和 Rust 接受本地路径或 HTTP(S) URL；本地路径直接引用原文件，远程内容保存在客户端存储根目录中。

一次调用只能包含本地来源或远程来源，不能混合。模型分片和视觉投影器放在同一个来源列表中，运行时根据 GGUF 元数据识别角色并验证配对。本地文件被删除或修改后，对应的旧注册项会失效。

## 运行时与请求参数

保持各类参数的作用域清晰分离：

- 浏览器客户端参数（`wasmThreading`、运行时资源 URL、`browserCache`），必须在初始化 `new SippClient(...)` 时设置。
- `models.add` 的来源列表和进度、取消选项，用于模型注册阶段。
- 本地端点加载选项（模型 ID、浏览器后端偏好、`NativeRuntimeConfig`），用于端点注册阶段。
- 运行时配置组（`context`、`sampling`、`scheduler`、`cache`、`placement`、`multimodal`、`residency`、`observability`），定义端点稳定的运行行为。
- 请求参数（`maxTokens`、`temperature`、`topP`、`stop`、取消控制、`emitTokens`），传递给 `query`、`chat` 或 `embed`。
- 仅本地支持的请求参数（上下文键、语法约束、媒体输入、嵌入归一化），不应发往网关或云端服务商端点。

规范参数映射和字段分组见[运行时参数](../reference/runtime-options.md)。

## 线程与浏览器执行

浏览器本地推理始终在专用 Worker 中运行。每次激活模型都会创建新的 Worker 和 Wasm 实例。`wasmThreading: 'pthread'` 启用多线程 WASM 运行时，需浏览器支持 `SharedArrayBuffer` 并配置跨源隔离响应头。

内置浏览器运行时要求提供 COOP/COEP 响应头。应用无法提供这些响应头时，需要设置 `wasmThreading: 'single-thread'`，并提供自定义单线程 `moduleUrl` 和 `wasmUrl` 资源。

原生 Node.js、Python 和 Rust 端点可通过 `context.n_threads` 和 `context.n_threads_batch` 手动指定 CPU 线程数。除非有确切性能数据，否则建议留空使用默认值。

## 文本、嵌入与视觉

- Query 和 Chat 需要支持文本生成的模型。
- Embed 需要支持嵌入计算的模型或运行时。
- 视觉聊天需要文本/视觉多模态模型，架构有要求时提供投影器数据。
- 获取流式文本需设置 `emitTokens`，在接收最终响应前（或同时）消费返回的 Token 迭代器。
- GBNF 语法和媒体输入仅支持在本地端点请求中使用。

## 相关文档

- [运行时参数](../reference/runtime-options.md)
- [Browser 包](../packages/browser.md)
- [Node.js 包](../packages/node.md)
- [Python 包](../packages/python.md)
- [Rust 包](../packages/rust.md)
- [浏览器缓存](browser-caching.md)
