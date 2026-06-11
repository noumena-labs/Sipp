# Next.js

Node.js 环境下运行的 App Router 路由处理器中，使用 `cogentlm-server`。客户端组件或纯浏览器模块中，改用 `cogentlm`。

Next.js App Router 默认将页面和布局视为服务端组件。只有模块需要访问浏览器 API、组件状态、事件监听器或浏览器本地的 CogentLM 引擎时，才在文件顶部添加 `'use client'`。

## 兼容网关 Profile 路由

通过路由处理器可有效防止提供商凭证泄露到客户端。引入 `cogentlm-server` 的路由文件，请确保声明 `export const runtime = 'nodejs';`。

要把某个路由作为浏览器的 `kind: 'gateway'` 端点使用，该路由必须兼容网关的 Profile 格式。可利用 `cogentlm-server` 提供的网关 Profile 助手解析请求体，返回标准 JSON 或 SSE 响应。路由内部可继续向直接提供商端点发起请求。

代码示例中的 `OPENAI_API_KEY="<mock-openai-key>"` 仅为演示。实际部署时，请务必从服务器环境变量或密钥管理服务中读取密钥。

```ts
// app/api/cogent/query/route.ts
import {
  CogentClient,
  decodeGatewayQueryBody,
  gatewayErrorResponse,
  gatewayTextResponseBody,
  gatewayTextStreamResponse,
} from 'cogentlm-server';

export const runtime = 'nodejs';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

export async function POST(request: Request): Promise<Response> {
  try {
    const decoded = decodeGatewayQueryBody(await request.json());
    const client = new CogentClient();
    const endpoint = await client.add('provider', {
      kind: 'provider',
      provider: 'openai',
      model: decoded.target,
      apiKey: requiredEnv('OPENAI_API_KEY'),
    });
    const run = client.query({
      ...decoded.request,
      endpoint,
    });
    if (decoded.stream) {
      return gatewayTextStreamResponse(run);
    }
    return Response.json(
      gatewayTextResponseBody(decoded.target, await run.response),
    );
  } catch (error) {
    const response = gatewayErrorResponse(error);
    return Response.json(response.body, response.init);
  }
}
```

浏览器客户端通过 `client.add({ kind: 'gateway' })` 调用该路由时，不要让路由返回非标准的自定义 JSON 结构（如 `{ text }`）。对浏览器而言，该路由必须表现得像一个正规的 HTTP 网关端点，即使它由 Next.js 应用自行实现。服务端可灵活决定将请求转发给提供商、本地端点或独立网关。

服务吞吐量较高时，将端点配置逻辑抽离到纯服务端共享模块，在请求生命周期内复用客户端实例。绝不要在客户端组件中引入此模块。

## 流式传输路由处理

浏览器需要实时接收 Token 更新，同时服务器必须妥善保管提供商凭证时，使用路由处理器实现流式传输。

```ts
// app/api/cogent/stream/route.ts
import { CogentClient } from 'cogentlm-server';

export const runtime = 'nodejs';

const encoder = new TextEncoder();

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

export async function POST(request: Request): Promise<Response> {
  const { prompt } = await request.json() as { prompt?: string };
  if (prompt == null || prompt.trim() === '') {
    return Response.json({ error: 'prompt is required' }, { status: 400 });
  }

  const client = new CogentClient();
  const endpoint = await client.add('provider', {
    kind: 'provider',
    provider: 'openai',
    model: requiredEnv('OPENAI_MODEL'),
    apiKey: requiredEnv('OPENAI_API_KEY'),
  });
  const run = client.query({
    endpoint,
    prompt,
    emitTokens: true,
    options: { maxTokens: 128 },
  });

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const batch of run.tokens) {
          controller.enqueue(encoder.encode(batch.text));
        }
        await run.response;
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
    cancel() {
      run.cancel('client_disconnected');
    },
  });

  return new Response(stream, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
```

## 浏览器本地客户端组件

在浏览器中运行本地推理依赖浏览器原生 API，因此相关逻辑必须封装在客户端组件中。

```ts
// app/local-chat/LocalChat.tsx
'use client';

import { useState } from 'react';
import { CogentClient } from 'cogentlm';

export function LocalChat(): JSX.Element {
  const [text, setText] = useState('');

  async function run(prompt: string): Promise<void> {
    const client = new CogentClient();
    try {
      const endpoint = await client.add('default', {
        kind: 'local',
        source: '/models/model.gguf',
      });
      const response = await client.query(prompt, {
        endpoint,
        maxTokens: 64,
      }).response;
      setText(response.text);
    } finally {
      await client.close();
    }
  }

  return (
    <button type="button" onClick={() => void run('Explain local inference.')}>
      {text || 'Run'}
    </button>
  );
}
```

如果覆盖了默认的资源路径（`moduleUrl`、`wasmUrl`、`pthreadModuleUrl` 或 `pthreadWasmUrl`），请务必成对提供对应运行时的 JavaScript 和 WASM 资源 URL。并且，只有在应用正确配置了跨源隔离头并开启 `SharedArrayBuffer` 支持的情况下，才能设置 `wasmThreading: 'pthread'`。

## 混合模式客户端组件

你可以使用同一个 `CogentClient` 实例，同时注册一个浏览器本地端点和一个兼容网关 Profile 的同源路由端点。发起请求时，只需切换传入的端点引用即可，调用 `query` 的代码无需任何改动。

```ts
// app/hybrid-chat/HybridChat.tsx
'use client';

import { useState } from 'react';
import { CogentClient, type EndpointRef } from 'cogentlm';

type InferenceMode = 'local' | 'providerRoute';

export function HybridChat(): JSX.Element {
  const [mode, setMode] = useState<InferenceMode>('local');
  const [text, setText] = useState('');

  async function run(prompt: string): Promise<void> {
    const client = new CogentClient();
    try {
      const localEndpoint = await client.add('browser-local', {
        kind: 'local',
        source: '/models/model.gguf',
      });
      const providerRouteEndpoint = await client.add('app-route', {
        kind: 'gateway',
        target: 'gpt-5-mini',
        baseUrl: window.location.origin,
        routes: { query: '/api/cogent/query' },
        authentication: { kind: 'none' },
      });
      const endpoint: EndpointRef =
          mode === 'local' ? localEndpoint : providerRouteEndpoint;
      const response = await client.query(prompt, {
        endpoint,
        maxTokens: 64,
      }).response;
      setText(response.text);
    } finally {
      await client.close();
    }
  }

  return (
    <>
      <select
        value={mode}
        onChange={(event) => setMode(event.currentTarget.value as InferenceMode)}
      >
        <option value="local">Browser local</option>
        <option value="providerRoute">Provider route</option>
      </select>
      <button type="button" onClick={() => void run('Explain hybrid inference.')}>
        {text || 'Run'}
      </button>
    </>
  );
}
```

浏览器端的网关描述符需要完整的 `http` 或 `https` 格式的 `baseUrl`。调用同源的 Next 路由时，使用 `window.location.origin`，并通过 `routes` 对象覆盖具体路径，例如 `routes: { query: '/api/cogent/query' }`。`target` 值会发送到服务端路由，用作提供商的模型标识符。

## 独立网关模式

需要跨多个应用统一管理目标策略、共享凭证、托管本地模型、实施限流或采集监控指标时，部署独立的 CogentLM 网关。浏览器直接调用独立网关时，切勿将长效 Token 硬编码到客户端包中。最佳做法是通过 Next 路由向客户端颁发短期 Token，在客户端端点配置的 `valueProvider` 中获取并使用：

```ts
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: {
    kind: 'bearer',
    valueProvider: async () => {
      const response = await fetch('/api/cogent/token', { method: 'POST' });
      return await response.text();
    },
  },
});
```

## 参考链接

- [Next.js Server and Client Components](https://nextjs.org/docs/app/getting-started/server-and-client-components)
- [Next.js Route Handlers](https://nextjs.org/docs/app/getting-started/route-handlers)
- [Next.js Route Segment Config](https://nextjs.org/docs/app/api-reference/file-conventions/route-segment-config)
