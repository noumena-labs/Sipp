# 浏览器包

浏览器包发布名称为 `cogentlm`。提供 `CogentClient` 客户端，支持浏览器本地 GGUF 推理、网关调用、提供商描述符、Token 流式传输、基于 OPFS 的模型缓存以及浏览器运行时生命周期管理。

各平台共享的 `add`、`query`、`chat`、`embed` 见[API 概述](../api/)。

## 安装

```bash
npm install cogentlm
```

在浏览器环境中使用此包。服务器路由或 Node 服务改用 [`cogentlm-server`](node.md)。

## 适用场景

- 浏览器本地执行文本和视觉模型推理。
- 浏览器运行时中调度 WebGPU 或 CPU 任务。
- 利用 OPFS 实现本地模型缓存。
- 通过网关发起 query、chat、embedding 调用。
- 为演示应用构建角色和导演助手。

## 本地推理

```ts
import { CogentClient, type ChatMessage } from 'cogentlm';

const client = new CogentClient();
const endpoint = await client.add('default', {
  kind: 'local',
  source: '/models/model.gguf',
  options: {
    backend: 'webgpu',
    runtime: {
      context: { n_ctx: 2048 },
    },
  },
});

const messages: readonly ChatMessage[] = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain CogentLM in one sentence.' },
];

const run = client.chat(messages, {
  endpoint,
  emitTokens: true,
  maxTokens: 64,
  contextKey: 'browser-local',
});

let streamed = '';
for await (const batch of run.tokens) {
  streamed += batch.text;
}
const response = await run.response;
console.log(streamed || response.text);
await client.close();
```

如果提示词已经符合目标模型的模板格式，请使用 `query` 方法。关于 `query`、`chat` 和 `embed` 的区别，请参阅 [API 概述](../api#query---原始提示词生成)。

## 经由网关推理

当需要由独立服务器统一管理模型路径、提供商凭证、访问策略和监控指标时，使用网关端点。

```ts
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: {
    kind: 'bearer',
    valueProvider: getShortLivedGatewayToken,
  },
});
const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain gateway inference.' },
];

const run = client.chat(messages, {
  endpoint,
  maxTokens: 64,
});
```

浏览器应用应使用短期网关 Token，或通过自身应用服务器路由代理。不要在浏览器包中硬编码提供商凭证或长期网关 Token。

## 浏览器运行时选项

浏览器运行时通过 Emscripten 将 CogentLM 的 Rust WASM ABI 与 llama.cpp 及 ggml 连接起来。浏览器支持兼容适配器时，引擎优先使用 WebGPU 运行 GGUF 文本和视觉模型；WebGPU 不可用时自动回退到 CPU。首次下载模型或导入文件后，基于 OPFS 的模型缓存可加速后续加载。

该包在运行时自动解析其打包的 JavaScript 和 WASM 资源，通常无需手动覆盖资源 URL。只有应用需要精细控制浏览器的执行、存储或本地运行时行为时，才需要配置 `executionMode`、`wasmThreading`、`browserCache` 以及本地端点的 `options.runtime`。

`CogentClient` 选项、WebGPU 和后端选择、Worker 模式、pthread 要求及本地运行时配置组的详情，见[运行时选项](../reference/runtime-options.md)。

## 相关文档

- [网关](../gateway/README.md)
- [Next.js](frameworks/nextjs.md)
- [TanStack](frameworks/tanstack.md)
- [React And Vite](frameworks/vite-react.md)
- [本地推理](../guides/local-inference.md)
- [运行时选项](../reference/runtime-options.md)
- [提供商](../guides/providers.md)
- [浏览器缓存](../guides/browser-caching.md)
- [网关与混合推理](../guides/gateway-hybrid.md)
- [示例与演示](../examples-demos.md)
- [维护者源码构建](../maintainers/source-builds.md)
