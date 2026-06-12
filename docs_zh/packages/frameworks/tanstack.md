# TanStack

基于 TanStack 的应用通常按以下几种模式使用 Sipp：

- 利用 TanStack Start 服务端函数处理纯服务端推理、管理提供商凭证、配置本地模型路径、下发网关 Token，提供类型安全的 RPC 调用。
- 借助 TanStack Start 服务器路由，让浏览器端通过 `kind: 'gateway'` 将该路由作为网关端点调用。
- 搭配 TanStack Query，缓存查询结果，客户端按需通过查询键重新获取完整生成内容。

处理 Token 流式传输时，应显式维护组件状态或使用自定义 Hook 接收并追加内容。TanStack Query 更适合处理基于 Promise 的最终完整结果，而非逐步拼接零散 Token。

## TanStack Start 服务端函数

服务端函数运行在服务器上，可通过加载器、组件、Hook 或其他服务端函数调用。确保 `sipp-server` 库的导入、提供商密钥的获取和网关 Token 的管理仅局限于服务端函数内部。

示例代码中的 `OPENAI_API_KEY="<mock-openai-key>"` 仅为演示。实际部署时请务必从环境变量或安全密钥库中加载真实的凭证。

```ts
// src/server/sipp.ts
import { createServerFn } from '@tanstack/react-start';
import { SippClient } from 'sipp-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

export const querySipp = createServerFn({ method: 'POST' })
  .inputValidator((data: { prompt: string }) => data)
  .handler(async ({ data }) => {
    const client = new SippClient();
    const endpoint = await client.add('provider', {
      kind: 'provider',
      provider: 'openai',
      model: requiredEnv('OPENAI_MODEL'),
      apiKey: requiredEnv('OPENAI_API_KEY'),
    });
    const run = client.query({
      endpoint,
      prompt: data.prompt,
      options: { maxTokens: 128 },
    });
    const response = await run.response;
    return { text: response.text, usage: response.usage };
  });
```

必须像对待任何对外公开的 API 端点那样严格校验服务端函数的输入参数。服务端函数本质上是可以通过网络调用的端点，务必在其实现或前置中间件中加入身份认证和权限隔离逻辑。

服务端函数非常适合为应用提供强类型 RPC 接口，返回自定义结构（如 `{ text }`）。但不适用于作为浏览器端 `client.add({ kind: 'gateway' })` 调用的代理路由，因为该方法要求端点严格兼容官方网关的 HTTP Profile 格式。

## TanStack Start 路由

浏览器包需要将框架内部路由当作网关端点时，实现对应的服务器路由。该路由需接收网关标准的请求 Profile，按要求返回特定格式的响应。借助网关 Profile 助手可轻松解码浏览器发来的请求结构并格式化 JSON 或 SSE 响应，同时该路由仍能在服务端直接向提供商 API 转发执行。

```ts
// src/routes/api/sipp/query.ts
import { createFileRoute } from '@tanstack/react-router';
import {
  SippClient,
  decodeGatewayQueryBody,
  gatewayErrorResponse,
  gatewayTextResponseBody,
  gatewayTextStreamResponse,
} from 'sipp-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

export const Route = createFileRoute('/api/sipp/query')({
  server: {
    handlers: {
      POST: async ({ request }) => {
        try {
          const decoded = decodeGatewayQueryBody(await request.json());
          const client = new SippClient();
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
      },
    },
  },
});
```

上述路由提取浏览器请求 Profile 中的 `model` 字段作为提供商的具体模型标识，同时将敏感提供商凭据保留在服务端。向公网暴露此路由前，务必增加业务层面的认证拦截或模型标识白名单校验。

需要跨多个应用统一实施限流、管理模型目标策略、共享提供商凭证并汇聚指标时，考虑部署独立的 Sipp 网关服务。

## 配合 TanStack Query 获取完整响应

UI 组件只关心最终生成的完整文本内容并期望利用查询缓存时，结合使用 TanStack Query。

```ts
import { useQuery } from '@tanstack/react-query';
import { querySipp } from '../server/sipp';

export function Answer({ prompt }: { readonly prompt: string }): JSX.Element {
  const result = useQuery({
    queryKey: ['sipp-query', prompt],
    queryFn: () => querySipp({ data: { prompt } }),
    enabled: prompt.trim() !== '',
  });

  if (result.isPending) return <p>Loading...</p>;
  if (result.isError) return <p>{result.error.message}</p>;
  return <pre>{result.data.text}</pre>;
}
```

将提示词、模型目标以及其他可能改变生成结果、对用户可见的选项都加入到查询键的依赖数组中。

## 流式传输 Token

实现 Token 实时流式传输时，建立一个返回流数据的服务器路由或服务端函数，在组件内借助内部状态逐步追加接收到的数据块。

```ts
import { useState } from 'react';

export function StreamingAnswer(): JSX.Element {
  const [text, setText] = useState('');

  async function run(prompt: string): Promise<void> {
    setText('');
    const response = await fetch('/api/sipp/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    if (response.body == null) {
      throw new Error('streaming response body is missing');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      setText((current) => current + decoder.decode(value, { stream: true }));
    }
  }

  return (
    <button type="button" onClick={() => void run('Explain streaming.')}>
      {text || 'Run'}
    </button>
  );
}
```

## 浏览器包集成

浏览器端执行的组件逻辑都必须引入浏览器版本的 `sipp` 包。无论是启动浏览器本地 GGUF 推理，还是利用短期 Token 和同源代理路由发起网关调用，都遵循这一原则。

```ts
import { useState } from 'react';
import { SippClient } from 'sipp';

export function LocalAnswer(): JSX.Element {
  const [text, setText] = useState('');

  async function run(prompt: string): Promise<void> {
    const client = new SippClient();
    try {
      const endpoint = await client.add('browser-local', {
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

切勿在任何面向浏览器的模块中导入 Node 专用的 `sipp-server`。

## 浏览器混合端点调用

可在同一个浏览器端 `SippClient` 实例上同时注册本地和同源网关端点，发起请求时按需切换传入的端点引用。这样同源路由能在服务端真正调用外部提供商 API，同时对浏览器保持透明，提供符合网关 Profile 规范的接口体验。

```ts
import { useState } from 'react';
import { SippClient, type EndpointRef } from 'sipp';

type InferenceMode = 'local' | 'providerRoute';

export function HybridAnswer(): JSX.Element {
  const [mode, setMode] = useState<InferenceMode>('local');
  const [text, setText] = useState('');

  async function run(prompt: string): Promise<void> {
    const client = new SippClient();
    try {
      const localEndpoint = await client.add('browser-local', {
        kind: 'local',
        source: '/models/model.gguf',
      });
      const providerRouteEndpoint = await client.add('app-route', {
        kind: 'gateway',
        target: 'gpt-5-mini',
        baseUrl: window.location.origin,
        routes: { query: '/api/sipp/query' },
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

浏览器端网关描述符需要 `http` 或 `https` `baseUrl`。同源 TanStack 路由调用时，传入 `window.location.origin` 并覆写对应的路由映射字典，例如 `routes: { query: '/api/sipp/query' }`。服务器路由会提取代码中的 `target` 值，用作调用实际提供商模型时的名称。

## 参考链接

- [TanStack Start Server Functions](https://tanstack.com/start/latest/docs/framework/react/guide/server-functions)
- [TanStack Start Server Routes](https://tanstack.com/start/latest/docs/framework/react/guide/server-routes)
- [TanStack Query useQuery](https://tanstack.com/query/latest/docs/framework/react/reference/useQuery)
