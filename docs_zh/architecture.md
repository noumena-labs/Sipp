# 架构

CogentLM 将推理与协议及部署解耦。公共 API 由底层 crate 组合而成，核心推理层专注推理逻辑，不涉及 HTTP 路由、序列化协议或部署方式。

## 基础 Crate

- `crates/sys`：unsafe FFI 绑定和 llama.cpp 原生胶水代码。
- `crates/core`：底层通用共享类型。
- `crates/engine`：管理本地推理、调度、生命周期和内存。
- `crates/shard`：GGUF 缓存规划与分片文件工具。
- `crates/client`：类型安全的端点注册，分发 query、chat、embed 请求。
- `crates/gateway-core`：不依赖特定协议的网关执行接口和管道排序。
- `crates/providers`：显式选择的外部服务商适配器。

## 公共库

- `lib/rust`：Rust 门面 Crate。
- `lib/web`：浏览器包源码。
- `lib/node`：Node.js 服务端包源码。
- `lib/python`：Python 包源码。
- `lib/gateway`：无路由的 HTTP 网关工具包。

## 应用与示例

- `apps/gateway-server`：开箱即用的官方网关应用。
- `apps/cli`：命令行本地推理应用。
- `examples`：可直接复用的精简集成示例。
- `demos`：基于公共接口构建的浏览器演示。
- `xtask`：构建、测试、运行、打包和维护的编排工具。

网关的分层架构见[网关架构](gateway/architecture.md)。
