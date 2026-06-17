# 本地推理

本地推理指在当前浏览器、Node.js、Python、Rust 或 CLI 进程内直接运行 GGUF 模型。应用完全掌控模型选择、运行时生命周期、资源清理和向用户暴露的请求选项。

通过 `SippClient.add` 注册本地端点，保存返回的引用，在调用 `query`、`chat` 或 `embed` 时传入该引用。

## 端点调用流程

1. 选择一个支持所需功能的 GGUF 模型。
2. 用本地描述符注册模型。
3. 在端点描述符中配置加载时的运行时选项。
4. 在调用 `query`、`chat`、`embed` 时传入生成选项。
5. 流式接收 Token 或等待最终结果。
6. 页面、Worker、服务或脚本不再需要模型时，关闭客户端释放资源。

本地端点不会自动路由。一个客户端可注册多个端点，使用时传入对应的引用。

## 模型来源

浏览器本地端点支持加载：

- 应用提供的模型 URL。
- 用户手动选择的本地文件。
- 多个分片 URL 或文件。
- 浏览器模型管理 API 返回的已安装模型 ID。
- 视觉模型的"模型+投影器"组合。

Node.js、Python、Rust 和 CLI 的本地端点使用文件系统路径。通过签出的代码库运行示例或冒烟测试时，可直接使用 `.build/models` 下缓存的测试模型。

## 运行时与请求参数

保持各类参数的作用域清晰分离：

- 浏览器客户端参数（`executionMode`、`wasmThreading`、运行时资源 URL、`browserCache`），必须在初始化 `new SippClient(...)` 时设置。
- 本地端点加载选项（模型来源、浏览器后端偏好、进度回调、`NativeRuntimeConfig`），用于端点注册阶段。
- 运行时配置组（`context`、`sampling`、`scheduler`、`cache`、`placement`、`multimodal`、`residency`、`observability`），定义端点稳定的运行行为。
- 请求参数（`maxTokens`、`temperature`、`topP`、`stop`、取消控制、`emitTokens`），传递给 `query`、`chat` 或 `embed`。
- 仅本地支持的请求参数（上下文键、语法约束、媒体输入、嵌入归一化），不应发往网关或云端服务商端点。

规范参数映射和字段分组见[运行时参数](../reference/runtime-options.md)。

## 线程与浏览器执行

浏览器的执行环境包含两个独立选项：

- `executionMode: 'worker'` 或 `auto`：环境支持 Worker 时，将推理计算移出 UI 主线程。
- `wasmThreading: 'pthread'`：启用多线程 WASM 运行时，需浏览器支持 `SharedArrayBuffer` 并配置跨源隔离响应头。

应用无法提供 COOP/COEP 响应头时，使用 `wasmThreading: 'single-thread'`。`executionMode: 'main-thread'` 通常仅用于调试或受限宿主环境。

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
