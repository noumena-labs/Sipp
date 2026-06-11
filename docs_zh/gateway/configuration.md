# 网关配置

`apps/gateway-server` 只通过一个 TOML 文件配置。源码/可执行文件环境与 Docker 环境使用相同的 TOML 结构，区别仅在于路径和绑定地址的解析方式。源码/可执行文件运行说明见[网关服务器](server.md)，容器运行说明见 [Docker](docker.md)。

## 示例配置

```toml
public_bind = "0.0.0.0:8080"
management_bind = "0.0.0.0:9090"
max_request_bytes = 1048576
max_concurrent_requests = 4
allowed_origins = []
admin_password_env = "COGENTLM_GATEWAY_ADMIN_PASSWORD"

[security.client_ip]
source = "peer"
trusted_proxy_cidrs = []

[security.rate_limit]
enabled = false
requests_per_minute = 60
burst = 60

[routes]
query = "/v1/query"
chat = "/v1/chat"
embed = "/v1/embed"
index = "/"
health = "/healthz"
readiness = "/readyz"
metrics = "/metrics"
admin = "/admin"

[[tokens]]
env = "COGENTLM_GATEWAY_TOKEN"
caller = "production-client"
targets = ["local"]

[[targets]]
name = "local"
type = "local"
model = "/models/model.gguf"
backend = "auto"
stats = "basic"
```

## 网关部署形态

网关通过相同的 TOML 架构支持三种部署形态。根据 `targets` 配置选择合适的模式。

### 本地 GPU 推理

网关服务器需要管理模型加载和 GPU 推理时，配置本地 GGUF 目标：

```toml
[[tokens]]
env = "COGENTLM_GATEWAY_TOKEN"
caller = "gpu-client"
targets = ["local-gpu"]

[[targets]]
name = "local-gpu"
type = "local"
model = "/models/model.gguf"
backend = "auto"
stats = "basic"
```

`backend` 设为 `"auto"`，或显式指定 `cuda`、`metal`、`vulkan` 等 GPU 后端。网关进程必须具有该 GGUF 路径的读取权限。Docker 中通常将主机的模型目录挂载到容器的 `/models`。

### 纯服务商路由

网关只需管理服务商凭证并将请求路由到上游 API，无需加载本地模型时，只配置服务商目标：

```toml
[[tokens]]
env = "COGENTLM_GATEWAY_TOKEN"
caller = "provider-client"
targets = ["openai-chat"]

[[targets]]
name = "openai-chat"
type = "openai"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"
timeout_seconds = 60
```

纯服务商配置没有 `type = "local"` 的目标、文件系统路径 `model` 和 `backend` 字段。此模式下网关不执行本地推理，适用 CPU 版本网关。

### 混合模式

客户端需要在本地模型和外部服务商之间选择时，同时配置两种目标：

```toml
[[tokens]]
env = "COGENTLM_GATEWAY_TOKEN"
caller = "hybrid-client"
targets = ["local-gpu", "openai-chat"]

[[targets]]
name = "local-gpu"
type = "local"
model = "/models/model.gguf"
backend = "auto"
stats = "basic"

[[targets]]
name = "openai-chat"
type = "openai"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"
timeout_seconds = 60
```

客户端通过 `model` 字段（如 `local-gpu` 或 `openai-chat`）指定公开的目标名称。

## 字段

| 字段 | 含义 |
| --- | --- |
| `public_bind` | 公共推理路由的监听地址。源码/可执行文件模式直接绑定在主机；Docker 模式绑定在容器内。 |
| `management_bind` | 健康检查、就绪检查、指标、首页和管理路由的监听地址。必须与 `public_bind` 不同。 |
| `max_request_bytes` | HTTP 请求体最大字节数，必须大于零。 |
| `max_concurrent_requests` | 可选全局并发请求上限。留空不限制。 |
| `allowed_origins` | 公共监听器的浏览器 CORS 允许源列表。空数组禁用 CORS。 |
| `admin_password_env` | 包含管理面板密码的环境变量名。必填且对应值不能为空。 |
| `security` | 客户端识别和速率限制设置（基于内存）。 |

`check` 命令只验证字段格式，不读取密钥、加载模型、连接服务商或绑定端口。

## 敏感密钥

TOML 文件只保存密钥的环境变量名。将实际机密存放在独立 `.env` 文件或生产环境的机密管理器中，切勿硬编码在 TOML 文件里。

```bash
COGENTLM_GATEWAY_ADMIN_PASSWORD=replace-me
COGENTLM_GATEWAY_TOKEN=replace-me
OPENAI_API_KEY=replace-me
ANTHROPIC_API_KEY=replace-me
```

在启动时，`serve` 命令会检查并拒绝缺失或为空的密钥环境变量。Bearer Token 的值严禁包含空格。

## 路由

`query`、`chat` 和 `embed` 是必需的公共路由。其余均为管理路由：

- `index`：可选的管理页 JSON 路由。
- `health`：可选的存活探针路由，响应 `ok`。
- `readiness`：可选的就绪探针路由，响应 `ready`。
- `metrics`：可选的 Prometheus 文本路由。
- `admin`：可选的管理面板路由。会话 JSON 端点位于 `<admin>/api/session`。

所有路由必须为绝对路径，且不能带有查询字符串或片段标识符。公共路由与管理路由内部均不能出现重复路径。

## Token

每个 `[[tokens]]` 块将一个 Bearer Token 环境变量映射到调用方标签和允许访问的目标列表：

```toml
[[tokens]]
env = "COGENTLM_GATEWAY_TOKEN"
caller = "browser-client"
targets = ["local", "openai-chat"]
```

- `env`：包含 Bearer Token 的环境变量名。
- `caller`：附加在请求元数据和诊断信息中的固定标签。
- `targets`：允许访问的 `[[targets]].name` 列表。留空则允许访问所有已配置的目标。

Token 值必须非空且不含空格。网关仅在 `serve` 启动阶段读取。

## 内存安全控制

当前版本中，网关的安全控制限于进程生命周期。服务器重启后，管理面板会话、CSRF Token、历史记录、客户端令牌桶、手动屏蔽列表和运行时配置覆盖都将丢失。网关不向 TOML 写入数据，不创建状态文件，不依赖外部缓存或数据库持久化。

内置示例根据 TCP 对端地址提取客户端 IP：

```toml
[security.client_ip]
source = "peer"
trusted_proxy_cidrs = []
```

`source` 支持 `peer`、`x_forwarded_for`、`x_real_ip`。只有网关部署在能保留真实客户端 IP 的可信反向代理后，并在 `trusted_proxy_cidrs` 中配置了代理 CIDR 时，这些请求头才生效。否则保持 `source = "peer"`。

每个客户端的速率限制需要显式配置：

```toml
[security.rate_limit]
enabled = false
requests_per_minute = 60
burst = 60
```

启用后，限流器以解析出的客户端 IP 为键名分配内存令牌。`requests_per_minute` 控制令牌填充速度，`burst` 控制并发峰值最大容量。

## 推理目标

每个 `[[targets]]` 块均会使用固定的名称发布一个模型或服务商端点。

### 本地 GGUF

```toml
[[targets]]
name = "local"
type = "local"
model = ".build/models/qwen2.5-0.5b-instruct-q4_0.gguf"
backend = "auto"
stats = "basic"
```

- `model`：进程有访问权限的 GGUF 路径。相对路径基于工作目录解析。
- `backend`：支持 `auto`、`cpu`、`cuda`、`metal`、`vulkan`。
- `stats`：支持 `off`、`basic`、`profile`。
- `runtime`：高级原生运行时设置（共享运行时选项架构）。

本地推理推荐 `backend = "auto"` 或明确指定 GPU 后端。`auto` 会按优先级选择可用的最佳编译后端：CUDA、Metal、Vulkan、CPU。指定 `cpu` 会禁用 GPU offload，仅用于诊断。显式指定的 GPU 后端未编译或不受支持时，进程报错退出。

`stats = "off"` 关闭运行时指标与后端分析。
`stats = "basic"` 仅开启运行时指标。`stats = "profile"` 同时开启指标与分析。

### OpenAI

```toml
[[targets]]
name = "openai-chat"
type = "openai"
model = "provider-model"
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"
timeout_seconds = 60
```

`base_url` 和 `timeout_seconds` 可选。启动 `serve` 时，网关从 `api_key_env` 指定的环境变量读取 API 密钥。

### OpenAI 兼容服务商

```toml
[[targets]]
name = "compatible-chat"
type = "openai_compatible"
model = "served-model"
base_url = "https://provider.example/v1"
token_env = "PROVIDER_TOKEN"
correlation_header = "x-request-id"
timeout_seconds = 60
```

`base_url` 和 `token_env` 必填。`correlation_header` 和 `timeout_seconds` 可选。

### Anthropic

```toml
[[targets]]
name = "anthropic-chat"
type = "anthropic"
model = "provider-model"
api_key_env = "ANTHROPIC_API_KEY"
version = "2023-06-01"
timeout_seconds = 60
```

`base_url`、`version`、`timeout_seconds` 可选。启动 `serve` 时，网关从 `api_key_env` 指定的环境变量读取 API 密钥。

## 绑定行为

源码/可执行文件模式直接将 `public_bind` 和 `management_bind` 绑定在主机。Docker 模式绑定在容器内部接口，通过 Compose `ports` 控制是否向主机暴露。

Docker 部署中：

- 网关进程监听容器接口（如 `0.0.0.0:8080` 和 `0.0.0.0:9090`）。
- 本地开发时，Compose 将两个端口都映射到 `127.0.0.1`。
- 生产环境默认公开公共端口，管理端口仅保留在 `127.0.0.1` 供本地主机访问。
- 本地模型路径需与 Compose 中容器卷的挂载点一致。
- 纯服务商模式 Docker 无需挂载模型，因为不加载本地 GGUF 目标。

## 管理面板

管理面板只在管理端口上运行。使用 `admin_password_env` 指定的环境变量值作为凭证，依赖极短有效期的 HTTP-only 会话 Cookie 鉴权，不在页面渲染任何密码、Bearer Token 或服务商密钥。

管理面板使用网关发布版中 `admin-ui` 目录下的 React 单页面应用，通过 `<admin>/api/*` 提供受会话保护的 JSON 端点。用 `POST <admin>/api/session` 登录，`DELETE <admin>/api/session` 登出。所有有副作用的 API 请求必须在 `x-cogentlm-admin-csrf` 请求头中附带 CSRF Token。面板中修改的运行时设置仅作用于当前进程，重启后恢复默认状态。
