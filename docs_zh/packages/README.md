# 使用核心库

Sipp 的所有公开包提供一致的面向端点的客户端模型。各平台共享的 `SippClient.add`、`query`、`chat`、`embed` 契约，以及端点描述符和网关-客户端对称模式，见[API 概述](../api)。

多数开发者应从这里开始，而非从源码构建。

## 包接口

| 接口 | 安装方式 | 主要用途 |
| --- | --- | --- |
| [API 概述](../api) | — | 跨平台共享的 `add`、`query`、`chat`、`embed` 接口。 |
| [Browser](browser.md) | `npm install @sipp/sipp` | 浏览器本地 GGUF 推理、WebGPU/WASM 运行时、浏览器端网关客户端。 |
| [Node.js](node.md) | `npm install @sipp/sipp-server` | Node 服务器进程、路由处理器、后端服务。 |
| [Python](python.md) | `pip install sipppy` | Python 服务、脚本、网关客户端。 |
| [Rust](rust.md) | `cargo add sipp-rs` | Rust 应用和服务。 |
| [Gateway Server](../gateway/server.md) | 目前需源码构建 |  HTTP 网关，支持本地和服务商目标。 |
| [Gateway Docker](../gateway/docker.md) | 基于源码的 Docker 构建 | 网关服务器的本地和生产环境容器工作流。 |
| [Gateway Toolkit](../gateway/toolkit.md) | 目前提供 Rust 源码包 | 构建自定义网关应用的 Rust 工具包。 |

当前发布工作流会发布浏览器 npm 包、Node npm 包、Python Wheel 包和 Rust 源码包。[Gateway](../gateway/) 章节将网关服务器作为面向用户的部署接口进行了说明，但尚未发布网关的二进制文件或公共镜像。

## 框架指南

如何在不同前端框架使用sipp，请参阅：

- [React and Vite](frameworks/vite-react.md)
- [Next.js](frameworks/nextjs.md)
- [TanStack](frameworks/tanstack.md)


## 参考资料

- [模型服务商](../guides/providers.md) — 模型提供商与网关的职责划分
- [运行时选项](../reference/runtime-options.md) — 运行选项层级映射与字段参考
- [源码构建](../maintainers/source-builds.md) — 基于当前代码库开发
