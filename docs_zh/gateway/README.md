# 网关

CogentLM 网关在本地 GGUF 和服务商前面架一层统一的 HTTP 边界。应用层代码写法不变：`CogentClient.add` 注册端点，获取引用，将引用传入 `query`、`chat`、`embed`进行推理。

网关需要一个独立进程来管理模型路径、服务商凭证、目标访问策略、并发限制、监控指标和运维路由。

## 注意事项

> [!WARNING]
> 目前正处于开发阶段, 网关服务器会频繁变动，可能不兼容旧版本。请时刻关注版本发布, 谨慎用于生产环境。加入 [Discord](https://discord.gg/abzgfghhrq) 了解最新进展。

## Getting Started

| 目的 | 看这里 |
| --- | --- |
| 服务器源码 | [服务器](server.md) |
| 构建和运行 Docker 镜像 | [Docker](docker.md) |
| TOML 配置文件说明 | [配置](configuration.md) |
| 用 curl、Postman 或裸 HTTP 测试 | [测试](testing.md) |
| 健康检查、指标、管理后台、入口 | [运维](operations.md) |
| 自己写网关应用 | [工具包](toolkit.md) |
| 了解各层的边界 | [架构](architecture.md) |
| 常见问题排查 | [故障排除](troubleshooting.md) |

当前发布流程会发包（npm、PyPI、crates.io），但不发独立的 gateway-server 二进制和容器镜像。部署官方服务器目前请使用源码或用 Dockerfile 自行构建。

## 几种网关形态

- **服务器**：`apps/gateway-server`，带 TOML 配置、Bearer Token 策略、本地和服务商目标、管理路由、指标和管理面板。
- **Docker 镜像**：`apps/gateway-server/Dockerfile`，运行 `cogentlm-gateway serve --config /etc/cogentlm/gateway.toml`。
- **网关工具包**：`lib/gateway`，提供编解码器、HTTP 错误辅助函数、鉴权和可观测性接口，以及官方 JSON/SSE 协议。
- **网关客户端**：浏览器、Node、Python、Rust 包都通过 `.add` 注册网关端点，跟本地端点、服务商端点一样。

## 部署形态

- **本地 GPU 推理**：配本地 GGUF 目标，用 `vulkan`、`cuda` 或 `metal` 跑网关，模型路径要挂给网关进程读。
- **纯服务商路由**：只配外部服务商目标（`openai`、`openai_compatible`、`anthropic`）。不需要模型路径，推理在服务商那边跑，用 CPU 版网关就够。
- **混合模式**：同时配本地 GPU 目标和服务商目标。客户端在请求的 `model` 字段里填公开的网关目标名。

## 默认路由

官方服务器示例默认用这些路由：

- 公开：`/v1/query`、`/v1/chat`、`/v1/embed`
- 管理：`/`、`/healthz`、`/readyz`、`/metrics`、`/admin`

这些路径是应用层的配置，不是核心库的行为。自定义网关可以自己决定路由。
