# Gateway Testing

Use this page when testing the first-party gateway with curl, Postman, or any
other raw HTTP client. The examples assume the default routes from
`apps/gateway-server/config/*.toml`.

## Environment

Bash:

```bash
export GATEWAY_URL="http://127.0.0.1:8080"
export GATEWAY_MANAGEMENT_URL="http://127.0.0.1:9090"
export SIPP_GATEWAY_TOKEN="replace-me"
export SIPP_GATEWAY_TARGET="local"
```

PowerShell:

```powershell
$env:GATEWAY_URL = "http://127.0.0.1:8080"
$env:GATEWAY_MANAGEMENT_URL = "http://127.0.0.1:9090"
$env:SIPP_GATEWAY_TOKEN = "replace-me"
$env:SIPP_GATEWAY_TARGET = "local"
```

## Management Probes

Health and readiness do not require bearer authentication:

```bash
curl --fail --silent "$GATEWAY_MANAGEMENT_URL/healthz"
curl --fail --silent "$GATEWAY_MANAGEMENT_URL/readyz"
curl --fail --silent "$GATEWAY_MANAGEMENT_URL/metrics"
```

The Admin Dashboard is available at:

```text
http://127.0.0.1:9090/admin
```

Log in with the value of the env var named by `admin_password_env` in TOML.

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

Finite text responses use JSON:

```json
{
  "id": "response",
  "model": "local",
  "text": "A gateway centralizes inference behind an HTTP boundary.",
  "finish_reason": "stop"
}
```

When usage is available, the response also includes:

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

Chat uses the same finite text response shape as query. Valid message roles are
`system`, `user`, and `assistant`.

## Embeddings

```bash
curl -sS "$GATEWAY_URL/v1/embed" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SIPP_GATEWAY_TARGET"'",
    "input": "gateway inference"
  }'
```

Embedding responses use JSON:

```json
{
  "id": "response",
  "model": "local",
  "embedding": [0.0123, -0.0456]
}
```

Embedding requires a target that supports embeddings. Text-only local models or
provider targets can return an execution error for `/v1/embed`.

## Streaming

Query and chat support server-sent events when the request contains
`"stream": true`:

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

The stream content type is `text/event-stream`. Events are newline-delimited
SSE frames:

```text
event: token
data: {"text":"Gateways","sequence":0}

event: usage
data: {"input_tokens":8,"output_tokens":9,"total_tokens":17}

event: done
data: {"finish_reason":"stop"}
```

If an error happens after streaming has started, the stream emits:

```text
event: error
data: {"error":{"code":"execution","message":"..."}}
```

## Postman

Create a Postman environment with these variables:

| Variable | Example |
| --- | --- |
| `gateway_url` | `http://127.0.0.1:8080` |
| `management_url` | `http://127.0.0.1:9090` |
| `gateway_token` | `replace-me` |
| `gateway_target` | `local` |

For public routes:

- Method: `POST`.
- Authorization: Bearer Token with `{{gateway_token}}`.
- Header: `Content-Type: application/json`.
- Body: raw JSON.
- Query URL: `{{gateway_url}}/v1/query`.
- Chat URL: `{{gateway_url}}/v1/chat`.
- Embed URL: `{{gateway_url}}/v1/embed`.

For management probes:

- Method: `GET`.
- URLs: `{{management_url}}/healthz`, `{{management_url}}/readyz`, and
  `{{management_url}}/metrics`.
- No bearer token is required.

Postman can display finite JSON responses directly. For streaming requests,
use a client that preserves SSE frames, such as `curl -N`, when debugging token
timing and terminal events.

## Common HTTP Failures

| Status | Common cause |
| --- | --- |
| `400` | Invalid JSON, invalid route body, or unsupported request field value. |
| `401` | Missing bearer token or malformed `Authorization` header. |
| `403` | Bearer token is valid but not allowed to use the requested target. |
| `404` | Requested `model` target is not configured. |
| `413` | Request body exceeds `max_request_bytes`. |
| `429` | `max_concurrent_requests` admission limit is full. |
| `500` | Target load or execution failure. Check gateway logs and target config. |

Non-streaming errors use JSON:

```json
{
  "error": {
    "code": "authorization",
    "message": "token is not allowed to access target"
  }
}
```
