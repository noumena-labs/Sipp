# 框架指南

本指南介绍如何在主流应用框架中集成 Sipp 的 JavaScript 包。

浏览器推理或浏览器代码中发起网关调用时，使用浏览器包 `@sipphq/sipp`。Node.js 运行时的纯服务端代码（路由处理器、服务端函数、API 路由、后台工作进程等）只使用 Node 包 `@sipphq/sipp-server`。

## 指南列表

- [React And Vite](vite-react.md)：浏览器本地配置、WebGPU/WASM 资源加载、OPFS 模型缓存、本地开发 HTTP 头配置。
- [Next.js](nextjs.md)：App Router 提供商调用路由、客户端组件集成、网关 Profile 路由兼容性及流式传输。
- [TanStack](tanstack.md)：TanStack Start 服务端函数、API 路由、TanStack Query 状态管理。

## 包选择参考

| 环境 | 推荐包 | 适用场景 |
| --- | --- | --- |
| 浏览器组件 | `@sipphq/sipp` | 浏览器本地 GGUF 推理，或直接向网关发起请求。 |
| Node 服务端路由 | `@sipphq/sipp-server` | 调用直接提供商端点、服务器本地推理、作为网关客户端。 |
| 网关 Profile 路由 | `@sipphq/sipp-server` | 为浏览器 `kind: 'gateway'` 端点提供兼容代理路由。 |
| 网关客户端 | 两者皆可 | 浏览器应用用短期 Token 访问独立网关；服务端应用用安全密钥访问网关。 |

## 服务端路由

框架服务端负责管理凭证时，Next.js 和 TanStack 的服务端路由应直接调用提供商端点。在仅限服务端的代码中注册提供商：

```ts
const endpoint = await client.add('provider', {
  kind: 'provider',
  provider: 'openai',
  model: requiredEnv('OPENAI_MODEL'),
  apiKey: requiredEnv('OPENAI_API_KEY'),
});
```

文档与示例中的 `OPENAI_API_KEY="<mock-openai-key>"` 仅为演示。绝不要将真实的提供商密钥泄露到浏览器端的构建产物中。

## 网关路由与字段名

浏览器端配置网关描述符时需提供 `http` 或 `https` 格式的 `baseUrl`，通过 `routes: { query, chat, embed }` 覆盖具体路由路径。服务端代码用 `@sipphq/sipp-server` 调用网关时，Node 端网关描述符对应使用 `queryRoute`、`chatRoute`、`embedRoute`。

不要在浏览器打包资源中包含提供商凭证或长期网关 Token。浏览器应用访问网关时，下发短期访问 Token 或通过应用服务器路由进行安全代理。

将框架路由封装成兼容浏览器 `kind: 'gateway'` 调用的端点时，使用 `@sipphq/sipp-server` 提供的 `decodeGatewayQueryBody()`、`decodeGatewayChatBody()`、`decodeGatewayEmbedBody()` 等解析函数及响应助手函数。这些助手帮助路由专注于鉴权策略、目标转发、提供商映射和客户端生命周期，无需关注底层网关 Profile JSON 编码细节。
