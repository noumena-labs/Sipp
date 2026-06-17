# 网关测试

使用 curl、Postman 或其他原生 HTTP 客户端测试官方网关时参考本文档。以下示例默认基于 `apps/gateway-server/config/*.toml` 中的路由配置。

## 环境变量配置

Bash：

```bash
export GATEWAY_URL="http://127.0.0.1:8080"
export GATEWAY_MANAGEMENT_URL="http://127.0.0.1:9090"
export SIPP_GATEWAY_TOKEN="replace-me"
export SIPP_GATEWAY_TARGET="local"
```

PowerShell：

```powershell
$env:GATEWAY_URL = "http://127.0.0.1:8080"
$env:GATEWAY_MANAGEMENT_URL = "http://127.0.0.1:9090"
$env:SIPP_GATEWAY_TOKEN = "replace-me"
$env:SIPP_GATEWAY_TARGET = "local"
```

## 管理端探针

调用健康检查与就绪探针无需 Bearer 身份验证：

```bash
curl --fail --silent "$GATEWAY_MANAGEMENT_URL/healthz"
curl --fail --silent "$GATEWAY_MANAGEMENT_URL/readyz"
curl --fail --silent "$GATEWAY_MANAGEMENT_URL/metrics"
```

管理面板的访问地址为：

```text
http://127.0.0.1:9090/admin
```

登录凭证为 TOML 中 `admin_password_env` 对应环境变量的值。

## Query

```bash
curl -sS "$GATEWAY_URL/v1/query" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -H "x-request-id: curl-query-1" \
  -d '{
    "model": "'"$SIPP_GATEWAY_TARGET"'",
    "prompt": "Explain gateway inference in one sentence.",
    "max_tokens": 64,
    "temperature": 0.2
  }'
```

非流式请求将返回 JSON 格式响应：

```json
{
  "id": "response",
  "model": "local",
  "text": "A gateway centralizes inference behind an HTTP boundary.",
  "finish_reason": "stop"
}
```

如果开启了 Token 使用量统计，响应还将包含：

```json
{
  "usage": {
    "input_tokens": 8,
    "output_tokens": 12,
    "total_tokens": 20
  }
}
```

## Chat

```bash
curl -sS "$GATEWAY_URL/v1/chat" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SIPP_GATEWAY_TARGET"'",
    "messages": [
      { "role": "system", "content": "Answer briefly." },
      { "role": "user", "content": "What does the gateway own?" }
    ],
    "max_tokens": 64
  }'
```

对话响应格式与 Query 相同。合法角色名称为 `system`、`user`、`assistant`。

## 向量嵌入 (Embeddings)

```bash
curl -sS "$GATEWAY_URL/v1/embed" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SIPP_GATEWAY_TARGET"'",
    "input": "gateway inference"
  }'
```

向量嵌入返回 JSON 格式响应：

```json
{
  "id": "response",
  "model": "local",
  "embedding": [0.0123, -0.0456]
}
```

调用嵌入接口需要目标模型支持向量生成。对仅支持文本生成的本地模型或服务商发起 `/v1/embed` 请求时，系统返回执行错误。

## 流式传输

若请求中设置 `"stream": true`，Query 和 Chat 会支持服务器发送事件（SSE）：

```bash
curl -N -sS "$GATEWAY_URL/v1/query" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SIPP_GATEWAY_TARGET"'",
    "prompt": "Write one short sentence about gateways.",
    "max_tokens": 64,
    "stream": true
  }'
```

此时的响应内容类型为 `text/event-stream`。事件会以换行符分隔的 SSE 帧形式返回：

```text
event: token
data: {"text":"Gateways","sequence":0}

event: usage
data: {"input_tokens":8,"output_tokens":9,"total_tokens":17}

event: done
data: {"finish_reason":"stop"}
```

如果流式传输过程中发生错误，输出如下：

```text
event: error
data: {"error":{"code":"execution","message":"..."}}
```

## Postman

可以创建一个配置了以下环境变量的 Postman 环境：

| 变量 | 示例 |
| --- | --- |
| `gateway_url` | `http://127.0.0.1:8080` |
| `management_url` | `http://127.0.0.1:9090` |
| `gateway_token` | `replace-me` |
| `gateway_target` | `local` |

测试公共路由：

- 请求方式：`POST`
- 身份验证：配置 Bearer Token 并填入 `{{gateway_token}}`
- 请求头：`Content-Type: application/json`
- 请求体：选择 raw 格式的 JSON
- Query 接口：`{{gateway_url}}/v1/query`
- Chat 接口：`{{gateway_url}}/v1/chat`
- Embed 接口：`{{gateway_url}}/v1/embed`

测试管理端探针：

- 请求方式：`GET`
- 接口地址：`{{management_url}}/healthz`、`{{management_url}}/readyz` 或 `{{management_url}}/metrics`
- 这些接口无需 Bearer Token 验证

Postman 可直接展示常规 JSON 响应，但调试流式输出或终端事件时，推荐使用能正确解析 SSE 帧的客户端（如 `curl -N`）。

## 常见 HTTP 错误

| 状态码 | 常见原因 |
| --- | --- |
| `400` | JSON 格式错误、请求体无效，或者含有不支持的字段值。 |
| `401` | 缺少 Bearer Token 或 `Authorization` 请求头格式错误。 |
| `403` | Bearer Token 有效，但没有权限访问指定的推理目标。 |
| `404` | 请求的 `model` 目标未在网关配置中声明。 |
| `413` | 请求体体积超过了 `max_request_bytes` 设定的上限。 |
| `429` | 并发请求数量超过了 `max_concurrent_requests`。 |
| `500` | 目标加载或执行失败。请检查网关运行日志与目标配置。 |

非流式请求发生错误时，将返回 JSON 响应：

```json
{
  "error": {
    "code": "authorization",
    "message": "token is not allowed to access target"
  }
}
```
