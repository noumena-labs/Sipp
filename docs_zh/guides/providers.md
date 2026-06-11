# 服务商支持

CogentLM 支持两种方式调用外部服务商：在可信服务端直接调用，或通过 CogentLM 网关间接调用。两种方式使用相同的端点模型：通过 `CogentClient.add` 注册，获取引用，传入 `query`、`chat`、`embed`。

服务商凭证必须存放在受信任的代码环境中。切勿将长期有效的服务商密钥打包到前端浏览器代码中。

## 直接服务商端点

当前服务端进程负责管理凭证生命周期和应用策略时，建议使用直接服务商端点。这也是 Next.js 和 TanStack 服务端代码推荐的路由模式。

```ts
import { CogentClient } from 'cogentlm-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

const client = new CogentClient();
const endpoint = await client.add('provider', {
  kind: 'provider',
  provider: 'openai',
  model: process.env.OPENAI_MODEL ?? 'gpt-5-mini',
  apiKey: requiredEnv('OPENAI_API_KEY'),
});

const run = client.chat({
  endpoint,
  messages: [{ role: 'user', content: 'Explain provider inference.' }],
  options: { maxTokens: 128, temperature: 0.2 },
});
console.log((await run.response).text);
```

> 文档和示例中仅使用 `OPENAI_API_KEY="<mock-openai-key>"` 等占位符。真实密钥始终存储在环境变量或机密管理器中。

## 服务商专属选项

对于通用的请求字段，请使用 CogentLM 标准的请求选项；针对特定服务商的专有字段，请将其置于 `providerOptions` 下：

```ts
const run = client.chat({
  endpoint,
  messages,
  options: { maxTokens: 128 },
  providerOptions: {
    reasoning_effort: 'low',
  },
});
```

`providerOptions` 仅适用于直接服务商端点。网关的扩展字段应放在 `endpointOptions` 或描述符级别的 `protocolOptions` 中，因为这些字段的解析取决于具体网关的实现。

## 支持服务商的网关目标

多个应用需要共享目标路由策略、凭证管理、本地模型服务、准入控制、指标监控，或需要一致 HTTP 边界时，使用官方网关。

OpenAI 目标：

```toml
[[targets]]
name = "openai-chat"
type = "openai"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"
```

兼容 OpenAI 协议的目标：

```toml
[[targets]]
name = "compatible-chat"
type = "openai_compatible"
model = "provider-model"
base_url = "https://provider.example/v1"
token_env = "COMPATIBLE_API_TOKEN"
correlation_header = "x-request-id"
```

Anthropic 目标：

```toml
[[targets]]
name = "anthropic-chat"
type = "anthropic"
model = "claude-3-5-sonnet-latest"
api_key_env = "ANTHROPIC_API_KEY"
```

使用这种方式，网关客户端只需获知公开目标名称、网关 URL 及网关认证令牌，真实的服务商凭证安全地留在网关进程内部。

## 浏览器应用

浏览器应用通常应调用业务后端或官方网关，而非直接向云端发送请求。如需实现 BYOK（自带密钥）工作流，可在运行时动态提供临时服务商密钥，并向用户明确提示侧信道安全风险。

## 相关文档

- [框架支持](../packages/frameworks/)
- [网关服务](../gateway/server.md)
- [网关与混合推理](gateway-hybrid.md)
- [运行时参数](../reference/runtime-options.md)
