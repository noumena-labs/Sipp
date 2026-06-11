# 网关架构

网关采用分层设计，各层职责独立。已移除的功能（自动路由装配、远程端点 API）不提供兼容层。

## 核心执行层

`crates/gateway-core` 只暴露强类型的 query、chat、embed 执行逻辑：

- `GatewayRequestContext` 和取消处理。
- `TargetResolver`、`Authorizer`、`AdmissionController`、`GatewayExecutor`。
- `GatewayPipeline` 负责执行顺序和准入许可的生命周期。
- 与协议无关的结果层和流式事件抽象层。

这一层不依赖 HTTP、Axum 路由、JSON、SSE、Bearer Token、状态码、别名、TOML 或任何固定限制。

`crates/client` 负责注册本地、服务商和网关三种端点。网关端点通过 HTTP 调用远程网关，必须显式指定，不会自动使用。

## 开发工具包

`lib/gateway` 提供一系列无路由的 HTTP 辅助组件，供自定义网关应用使用：

- `ProtocolCodec`：请求、响应、流和错误的传输格式抽象。
- `Authenticator`：自定义身份验证接口。
- `ErrorTranslator`：应用层 HTTP 错误映射。
- `GatewayCodec`：官方 Cogent JSON/SSE 协议实现。
- `GatewayHttpError` 及 SSE/错误响应编码器。

这套工具不注册路由、不暴露 Router 对象、不接管 handler。应用需自行解码请求、选择推理目标、调用 `client.query()` / `client.chat()` / `client.embed()`，然后编码响应。

## 公共端点

Rust、Node、Python、浏览器包通过相同的 `.add` 暴露网关端点描述符，与本地端点和提供商端点结构一致：

- 协议目标。
- 网关基础 URL。
- Query、chat、embed 的路由。
- 身份验证策略。
- 静态请求头。
- 超时策略。
- 协议特定请求选项。

端点 ID 只在调用 `.add` 时提供。本地模型、服务商和网关是三种不同的描述符，但只要拿到了端点引用，`query`、`chat`、`embed` 的调用方式完全一致。

## 官方应用

`apps/gateway-server` 是官方提供的开箱即用应用。其 Bearer Token、目标访问控制、并发限制、CORS、路由、管理端口、指标和 TOML 配置均由应用自行决定，非核心库强制要求。

`examples/gateway` 演示了标准的自定义开发流程：

- 创建 `CogentClient`。
- 用 `.add` 注册本地、服务商或网关端点。
- 在示例应用内定义 Axum 路由。
- 在各路由内解码请求体、选择端点、调用 `client.*`、编码响应。

`/v1/query`、`/v1/chat`、`/v1/embed` 这些默认路径只是应用层的约定。核心库提供编解码器和端点传输通道，不拥有路由。
