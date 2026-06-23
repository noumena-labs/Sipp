# 网关工具包

`sipp-gateway` 是无路由的 Rust HTTP 工具包，为希望通过自有服务器框架集成 Sipp 推理服务的应用而设计。

该工具包提供编解码器、身份验证和可观测性 Trait、HTTP 错误辅助函数以及官方 JSON/SSE 规范。应用可完全控制套接字绑定、路由注册、配置加载和部署策略。

如果只需要开箱即用、支持 TOML 配置、Bearer Token 鉴权、目标路由、指标监控和双监听器的独立服务，使用[网关服务器](server.md)。

## 分发方式

对应 Rust crate `sipp-gateway`。crates.io 发布 `sipp-rs` 与 `sipp-sys` 两个 crate；工具包本身有意以源码形式分发。直接依赖源码的方式见[源码构建](../maintainers/source-builds.md)。

## 适用场景

- 为应用定制专有 HTTP 网关路由。
- 将 HTTP 请求体转换为强类型 Sipp 请求结构。
- 对 JSON 和 SSE 响应进行标准编码。
- 保证与 Sipp 客户端间采用官方一致的协议规范。

## 极简处理器示例

```rust
use sipp_gateway::{GatewayCodec, ProtocolCodec};

let codec = GatewayCodec;
let mut decoded = codec.decode_query(&body)?;
decoded.request.endpoint = Some(resolve(&decoded.target)?);
let response = client.query(decoded.request).await?;
let bytes = codec.encode_text(&decoded.target, &response)?;
```

自定义网关应用需自行实现套接字管理、路由分发、鉴权、配置文件解析、目标策略、CORS、日志及默认部署配置。在 Node 框架中实现官方网关规范时，可使用 `@sipphq/sipp-server` 导出的配套辅助函数。

## 职责边界

`lib/gateway` 只提供辅助构件，不是完整应用：

- 不注册路由。
- 不绑定监听器。
- 不处理 Bearer Token 鉴权。
- 不处理 TOML 解析、CORS、指标收集或部署逻辑。

`/v1/query`、`/v1/chat`、`/v1/embed` 路径由使用该工具包的应用自行决定。

## 相关文档

- [架构](architecture.md)
- [网关与混合推理](../guides/gateway-hybrid.md)
- [开发框架](../packages/frameworks/README.md)
- [源码构建](../maintainers/source-builds.md)
