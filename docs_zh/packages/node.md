# Node.js 包

Node.js 包发布名称为 `@sipp/sipp-server`。为 Node 服务器进程、路由处理器和框架服务端函数提供 Sipp 客户端 API。使用该包时，应用全权负责管理框架的路由、请求校验、身份验证和部署策略。

各平台共享的 `add`、`query`、`chat`、`embed` 见[API 概述](../api)。

## 安装

```bash
npm install @sipp/sipp-server
```

仅在 Node 运行时代码中使用此包。浏览器组件改用 [`@sipp/sipp`](browser.md)。

## 适用场景

- 服务端本地执行模型推理。
- 服务端调用网关和提供商进行推理。
- Node 进程内 Token 流式传输。
- 构建 Node 运行时中的框架路由处理器。
- 选择原生绑定的推理后端。

## 本地推理 (Query)

```ts
import { SippClient } from '@sipp/sipp-server';

const client = new SippClient();
const endpoint = await client.add('default', {
  kind: 'local',
  modelPath: process.argv[2],
  config: {
    context: { n_ctx: 2048 },
    scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
    cache: { mode: 'live_slot_prefix' },
    observability: { runtime_metrics: true },
  },
});
const queryPrompt = [
  '<|system|>',
  'Answer concisely.',
  '<|user|>',
  'Explain Sipp in one sentence.',
  '<|assistant|>',
].join('\n');

const run = client.query({
  endpoint,
  // query 接收原始提示词；请确保提示词匹配目标模型的格式模板。
  prompt: queryPrompt,
  emitTokens: true,
  options: { maxTokens: 64, temperature: 0.7 },
  local: { contextKey: 'node-local' },
});

let streamed = '';
for await (const batch of run) {
  streamed += batch.text;
}
const response = await run.response;
console.log(streamed || response.text);
```

设置环境变量 `SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal` 来选择原生后端引擎。关于本地运行时的配置参数与请求选项说明，请参阅[运行时选项](../reference/runtime-options.md)。

## 网关推理

```ts
function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: requiredEnv('SIPP_GATEWAY_TARGET'),
  baseUrl: requiredEnv('SIPP_GATEWAY_URL'),
  authentication: {
    kind: 'bearer',
    value: requiredEnv('SIPP_GATEWAY_TOKEN'),
  },
});
const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain gateway inference.' },
];
const run = client.chat({
  endpoint,
  messages,
  options: { maxTokens: 64 },
});
console.log((await run.response).text);
```

应用程序只需要提供网关 URL、Bearer 凭证以及公开的目标名称。提供商凭证和本地模型路径均由网关进程负责管理。

## 客户端直接调用提供商

仅在受信任的服务端代码中使用提供商端点。将提供商的 API 密钥存储在服务器环境变量中；以下示例中的 `OPENAI_API_KEY="<mock-openai-key>"` 仅为演示。

```ts
function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

const endpoint = await client.add('provider', {
  kind: 'provider',
  provider: 'openai',
  model: process.env.OPENAI_MODEL ?? 'gpt-5-mini',
  apiKey: requiredEnv('OPENAI_API_KEY'),
});
const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain provider inference.' },
];
const run = client.chat({
  endpoint,
  messages,
  options: { maxTokens: 64 },
});
console.log((await run.response).text);
```

通过 `providerOptions` 可传递特定于该提供商的请求字段。关于提供商与网关的职责划分，请参阅[提供商](../guides/providers.md)。

## 网关 Profile 助手

Node 路由需要向浏览器的 `kind: 'gateway'` 客户端提供类似原生网关端点的接口时，可使用网关 profile 助手。这些函数负责解码 `model`、`prompt`、`messages`、`input` 字段及 snake_case 风格生成选项，然后构造 JSON 或 SSE 响应。路由可基于解码后的请求，自由选择向提供商、本地端点或独立网关发起执行。

```ts
import {
  SippClient,
  decodeGatewayQueryBody,
  gatewayErrorResponse,
  gatewayTextResponseBody,
  gatewayTextStreamResponse,
} from '@sipp/sipp-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

export async function handleQuery(request: Request): Promise<Response> {
  try {
    const decoded = decodeGatewayQueryBody(await request.json());
    const client = new SippClient();
    const endpoint = await client.add('provider', {
      kind: 'provider',
      provider: 'openai',
      model: decoded.target,
      apiKey: requiredEnv('OPENAI_API_KEY'),
    });
    const run = client.query({ ...decoded.request, endpoint });
    return decoded.stream
      ? gatewayTextStreamResponse(run)
      : Response.json(
          gatewayTextResponseBody(decoded.target, await run.response),
        );
  } catch (error) {
    const response = gatewayErrorResponse(error);
    return Response.json(response.body, response.init);
  }
}
```

对于处理 `/v1/chat` 和 `/v1/embed` 的路由，请分别使用 `decodeGatewayChatBody()` 和 `decodeGatewayEmbedBody()`。构建嵌入响应时请使用 `gatewayEmbeddingResponseBody()`。

## 框架路由集成

只在服务端代码中使用 `@sipp/sipp-server`。典型场景包括配置了 `runtime = 'nodejs'` 的 Next.js App Router 路由处理器、TanStack Start 服务端函数、Express 路由或后台工作进程。切勿将其引入浏览器包。

## 相关文档

- [网关服务器](../gateway/server.md)
- [Next.js](frameworks/nextjs.md)
- [TanStack](frameworks/tanstack.md)
- [本地推理](../guides/local-inference.md)
- [提供商](../guides/providers.md)
- [运行时选项](../reference/runtime-options.md)
- [网关与混合推理](../guides/gateway-hybrid.md)
- [维护者源码构建](../maintainers/source-builds.md)
