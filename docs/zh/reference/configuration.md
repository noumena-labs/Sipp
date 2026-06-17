# 配置

Sipp 的配置严格按职责划分。核心 Crate 不处理 HTTP 路由、身份验证方案、TOML 文件或部署策略。

## 运行时配置

本地运行时配置由端点描述符或包级别的运行时选项管理。常见配置项包括上下文大小、调度器行为、缓存模式、可观测性、采样和后端选择。共享选项的完整列表见[运行时选项](runtime-options.md)。

## 网关配置

`apps/gateway-server` 负责网关应用的 TOML 配置：

- `[routes]` 配置公共路由和管理路由。
- `admin_password_env` 指定包含管理面板密码的环境变量。
- `[[tokens]]` 将 Bearer Token 环境变量映射到调用方标签和允许访问的目标。
- `[[targets]]` 定义本地、OpenAI、兼容 OpenAI 或 Anthropic 目标。本地目标可选 `backend = "auto"`、`cpu`、`cuda`、`metal` 或 `vulkan`。完整 Schema 见[网关配置](../gateway/configuration.md)。

自定义传输格式、身份验证方案和路由布局应放在由 `lib/gateway` 组合而成的独立应用中。

## 环境变量

- `SIPP_GATEWAY_TOKEN`：开发环境或示例中使用的网关 Bearer Token。
- `SIPP_GATEWAY_ADMIN_PASSWORD`：网关示例中使用的管理面板密码。
- `SIPP_GATEWAY_URL`：客户端示例使用的网关基础 URL。
- `SIPP_NODE_BACKEND`：Node 运行时的计算后端选择。
- `SIPP_PYTHON_BACKEND`：Python 运行时的计算后端选择。
- `OPENAI_API_KEY`：OpenAI 示例及提供商支持的网关目标的凭证。
