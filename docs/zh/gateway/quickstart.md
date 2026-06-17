# 网关快速上手

经由网关, 我们用本地路径加载 GGUF 模型进行推理, 同时将请求路由到上游时用纯服务商路径。如需部署到生产环境, 请阅读[服务器](server.md)和 [Docker](docker.md) 文档。

## 从源码构建与运行：本地模式

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/config/local.toml.example apps/gateway-server/config/local.toml
```

编辑 `apps/gateway-server/config/local.toml`：

- 将本地目标的 `model` 指向工作区根目录可见的 GGUF 文件。
- 保持本地源绑定在 `127.0.0.1`。
- 除非修改了 `.env` 中的密钥名称，否则保留 `admin_password_env = "SIPP_GATEWAY_ADMIN_PASSWORD"`。

加载密钥并启动：

```bash
set -a
. apps/gateway-server/.env
set +a
sipp run gateway-server check --config apps/gateway-server/config/local.toml --backend vulkan
sipp run gateway-server serve --config apps/gateway-server/config/local.toml --backend vulkan
```

NVIDIA 主机将后端设为 `cuda`，macOS 主机设为 `metal`。

## 从源码构建与运行：纯服务商模式

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/config/provider-only.toml.example apps/gateway-server/config/provider-only.toml
```

在 `apps/gateway-server/.env` 中配置服务商密钥，然后运行：

```bash
set -a
. apps/gateway-server/.env
set +a
sipp run gateway-server check --config apps/gateway-server/config/provider-only.toml --backend cpu
sipp run gateway-server serve --config apps/gateway-server/config/provider-only.toml --backend cpu
```

在检出的纯服务商示例中，请求目标设为 `openai-chat`。

## Docker

Docker 部署需要一个仅含密钥的 `.env` 文件、一个网关 TOML 配置文件和一个明确的 Compose 文件：

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
cp apps/gateway-server/config/development.toml.example apps/gateway-server/config/development.toml
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up
```

纯服务商 Docker 部署使用 `development-provider-only.yml.example` 和 `config/provider-only.toml.example`。

## 首次 HTTP 请求

打开第二个终端窗口：

```bash
set -a
. apps/gateway-server/.env
set +a
export GATEWAY_URL="http://127.0.0.1:8080"
export GATEWAY_MANAGEMENT_URL="http://127.0.0.1:9090"

curl --fail --silent "$GATEWAY_MANAGEMENT_URL/readyz"
curl -sS "$GATEWAY_URL/v1/query" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","prompt":"Explain gateway inference.","max_tokens":64}'
```

纯服务商示例使用 `"model":"openai-chat"`。

打开 `http://127.0.0.1:9090/admin`，用 `SIPP_GATEWAY_ADMIN_PASSWORD` 的值登录。
