# React 与 Vite

React 和 Vite 是 `@sipphq/sipp` 浏览器包的基础集成环境。本指南介绍 Vite 配置方法、本地开发 HTTP 头设置、运行时资源覆写机制以及浏览器示例。

要全面了解本地推理选项的配置，请参阅[本地推理](../../guides/local-inference.md)和[运行时选项](../../reference/runtime-options.md)。

## 安装

```bash
npm install @sipphq/sipp
```

## 浏览器本地推理

仅在浏览器端代码中使用 `@sipphq/sipp`。本地端点的 `source` 可以是应用服务器提供的模型 URL、用户上传的 `File` 对象、已缓存的模型 ID，或者是分块下载的数据源。

```ts
import { useState } from 'react';
import { SippClient } from '@sipphq/sipp';

export function LocalQuery(): JSX.Element {
  const [text, setText] = useState('');

  async function run(): Promise<void> {
    const client = new SippClient();
    try {
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
      const response = await client.query('Explain Sipp.', {
        endpoint,
        maxTokens: 64,
      }).response;
      setText(response.text);
    } finally {
      await client.close();
    }
  }

  return (
    <button type="button" onClick={() => void run()}>
      {text || 'Run'}
    </button>
  );
}
```

省略 `backend` 时，浏览器运行时会选择后端引擎。如果 UI 需要明确请求 WebGPU 并由应用层处理后端错误，请显式设置 `backend: 'webgpu'`。

## Vite 配置

打包提供的 WASM 运行时使用 pthread，必须依赖 `SharedArrayBuffer` 和跨源隔离。使用默认浏览器运行时前，先配置 Vite 的开发与预览服务器请求头：

```ts
// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
```

应用无法配置这些响应头时，需要设置 `wasmThreading: 'single-thread'`，并提供自定义单线程 `moduleUrl` 和 `wasmUrl` 资源。仅在调试或受限主机环境中使用 `executionMode: 'main-thread'`。

## 运行时资源覆盖

浏览器包在运行时会自动解析内置的 Emscripten JavaScript 和 WASM 资源地址。大多数 Vite 应用直接使用 `new SippClient()` 即可，无需手动修改资源路径。

只有当打包工具或部署流程改变了资源存放位置时，才需要覆盖运行时资源 URL：

```ts
const client = new SippClient({
  moduleUrl: '/assets/sipp-wasm-pthread.js',
  wasmUrl: '/assets/sipp-wasm-pthread.wasm',
});
```

`moduleUrl` 和 `wasmUrl` 覆盖当前选择的运行时。默认选择 pthread。自定义单线程构建还必须设置 `wasmThreading: 'single-thread'`。

## 模型文件与缓存

应用可以直接提供模型文件的 URL，也可以让用户选取本地的 `.gguf` 文件。浏览器支持时，运行时会模型数据存储到 OPFS 中。首次下载或导入文件后，后续加载直接读取本地缓存。

通过 `SippClient` 上的 `browserCache` 选项调整浏览器缓存策略，或通过本地端点描述符的 `options.runtime` 调整本地运行时行为。详情见[浏览器缓存](../../guides/browser-caching.md)和[运行时选项](../../reference/runtime-options.md)。

## 官方示例

从源码仓库构建时，启动本地服务运行浏览器示例：

```bash
sipp run examples serve browser
```

访问终端输出的 URL 并打开以下页面：

- `/query.html`
- `/chat.html`
- `/embed.html`
- `/gateway_local.html`
- `/gateway_query.html`
- `/gateway_chat.html`
- `/gateway_embed.html`

其中网关示例页面演示了浏览器端调用网关 Profile 端点的方法。生产环境的服务端路由应部署在全栈框架、应用服务器或官方网关服务器中。

## 相关文档

- [浏览器包](../browser.md)
- [本地推理](../../guides/local-inference.md)
- [运行时选项](../../reference/runtime-options.md)
- [提供商](../../guides/providers.md)
- [网关服务器](../../gateway/server.md)
- [浏览器缓存](../../guides/browser-caching.md)
