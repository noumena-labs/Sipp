# Sipp 文档

Sipp 为浏览器、Node.js、Python、Rust 应用提供本地推理和网关推理运行时。整个项目基于统一的客户端模型：通过 `SippClient.add` 注册本地或远程端点，获取端点引用，然后传入 `query`、`chat` 或 `embed`。

本书从发布包开始介绍。源码构建、仓库架构和贡献流程位于"维护者"章节。

> [!WARNING]
> Sipp 会频繁更新。遇到问题、Bug 或想要新功能，欢迎在 [GitHub](https://github.com/noumena-labs/Sipp) 或 [Discord](https://discord.gg/abzgfghhrq) 提出。

## 从哪开始

- [路线图](roadmap.md) — 工程里程碑、内存架构和长期愿景
- [安装指南](getting-started/installation.md) — 各语言的安装命令
- [快速上手](getting-started/quickstarts.md) — 浏览器、Node.js、Python、Rust、网关的最简示例
- [使用核心库](packages/) — 各语言包的 API 详解
- [网关服务](gateway/) — 官方网关服务器、Docker 工作流、配置、测试、运维、工具包、架构
- [框架集成](packages/frameworks/) — Next.js、TanStack、React/Vite 的集成方式
- [网关与混合推理](guides/gateway-hybrid.md) — 什么时候用本地端点、网关端点、服务商端点
- [维护者指南](maintainers/) — 源码构建、测试、仓库结构、贡献流程

## 本地构建文档

在源码目录下执行：

```bash
sipp docs build
sipp docs serve
```

`sipp docs build` 会自动安装 `mdbook` 和 `mdbook-mermaid`，解压内置的 Mermaid JS 资源，输出文档到 `book/`。`sipp docs serve` 先执行同样的构建，然后启动支持热重载的本地服务。如果 `sipp` 未激活，使用 `cargo xtask docs ...` 代替。
