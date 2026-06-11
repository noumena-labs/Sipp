# 网关运维

官方网关有公共和管理两个监听器。部署时请在运维层面保持严格隔离。

## 公共监听器

公共监听器提供以下推理路由：

- `/v1/query`
- `/v1/chat`
- `/v1/embed`

所有发往公共监听器的请求必须携带有效 Bearer Token（由 `[[tokens]]` 策略验证）。请求中的 `model` 字段代表公开目标名称，网关将其解析为具体本地模型或外部服务商端点。

将公共监听器暴露在可信网络之外时，务必在前置配置 TLS、外部身份验证、速率限制和网络入口控制。

## 管理端监听器

管理端监听器提供以下路由：

- `/`：可选首页 JSON 路由。
- `/healthz`：存活探针路由，响应 `ok`。
- `/readyz`：就绪探针路由，响应 `ready`。
- `/metrics`：Prometheus 文本格式指标路由。
- `/admin`：受密码保护的管理面板。

务必保持管理端监听器私有。Docker 生产部署中，Compose 模板默认将管理端端口绑定到主机 `127.0.0.1`。

## 管理面板

管理面板的登录密码从 TOML 中 `admin_password_env` 指定的环境变量提取。面板使用极短有效期 HTTP-only 会话 Cookie 鉴权，不在页面渲染密码、Bearer Token 或服务商密钥。

通过管理面板可查看已配置的路由、目标、当前加载的本地推理后端和请求指标。切勿将管理面板暴露到公网。

## 指标

指标路由以 Prometheus 格式暴露指标，目前包含各操作维度的请求数和错误数。

```text
cogentlm_gateway_requests_total{operation="query"} 3
cogentlm_gateway_errors_total{operation="chat"} 1
```

本地运行时指标由 `stats` 配置决定：

- `off`：关闭运行时指标和后端分析。
- `basic`：仅开启运行时指标。
- `profile`：同时开启运行时指标和后端分析。

## 日志

网关通过 `tracing` 输出 JSON 格式日志。通过环境变量 `RUST_LOG` 控制日志级别：

```bash
RUST_LOG=info
RUST_LOG=debug,cogentlm_gateway_server=trace
```

切勿在日志中输出 Bearer Token 值、服务商凭证或生产环境 TOML 配置内容。

## CORS

`allowed_origins` 控制浏览器对公共监听器的跨域访问。空数组禁用 CORS。只添加受信任的源：

```toml
allowed_origins = ["https://app.example.com"]
```

浏览器客户端应使用运行时动态分发的短效网关 Token，而非将长期 Token 硬编码到前端构建产物中。

## 敏感密钥

网关依赖两类敏感密钥：

- `admin_password_env`：TOML 中配置的管理面板密码环境变量名。
- Token 和服务商密钥：变量名配在 TOML，真实值在 `serve` 启动时从进程环境读取。

妥善保管环境变量密钥文件，严禁提交到版本控制系统。条件允许时推荐使用部署平台原生的机密存储服务。
