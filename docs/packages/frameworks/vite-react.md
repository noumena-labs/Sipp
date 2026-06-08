# React And Vite

React and Vite are the baseline browser integration for the `cogentlm`
package. Use this guide for Vite-specific setup, local development headers,
runtime asset overrides, and the source browser examples.

For the full local inference option map, see
[Local Inference](../../guides/local-inference.md) and
[Runtime Options](../../reference/runtime-options.md).

## Install

```bash
npm install cogentlm
```

## Browser Local Query

Use `cogentlm` only in browser code. A local endpoint `source` can be a model
URL served by the app, a user-provided `File`, an installed model id, or shard
sources.

```ts
import { useState } from 'react';
import { CogentClient } from 'cogentlm';

export function LocalQuery(): JSX.Element {
  const [text, setText] = useState('');

  async function run(): Promise<void> {
    const client = new CogentClient();
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
      const response = await client.query('Explain CogentLM.', {
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

Omit `backend` to let the browser runtime choose a compatible backend. Use
`backend: 'webgpu'` when the UI should explicitly request WebGPU and surface
errors or fallbacks itself.

## Local Development Headers

The pthread WASM runtime requires `SharedArrayBuffer` and cross-origin
isolation. Configure Vite dev and preview headers before using
`wasmThreading: 'pthread'`:

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

Use `wasmThreading: 'single-thread'` when the app cannot serve those headers.
Use `executionMode: 'main-thread'` only for debugging or constrained hosts.

## Runtime Asset Overrides

The browser package resolves its packaged Emscripten JavaScript and WASM assets
at runtime. Most Vite apps can use `new CogentClient()` without asset
overrides.

Override runtime asset URLs only when your bundler or deployment moves package
assets:

```ts
const client = new CogentClient({
  moduleUrl: '/assets/cogentlm-wasm.js',
  wasmUrl: '/assets/cogentlm-wasm.wasm',
});
```

When overriding assets, provide both `moduleUrl` and `wasmUrl`. For pthread
runtime assets, provide both `pthreadModuleUrl` and `pthreadWasmUrl`.

## Model Files And Cache

Serve model URLs from the application or let users select local `.gguf` files.
The browser runtime stores model data through OPFS where available, so repeated
loads can stay local after the first import or fetch.

Tune browser storage with `browserCache` on `CogentClient` and tune local
runtime behavior with `options.runtime` on the local endpoint descriptor. See
[Browser Caching](../../guides/browser-caching.md) and
[Runtime Options](../../reference/runtime-options.md).

## Existing Examples

Serve the source examples when working from a checkout:

```bash
clm run examples serve browser
```

Then open the printed URL and use:

- `/query.html`
- `/chat.html`
- `/embed.html`
- `/gateway_local.html`
- `/gateway_query.html`
- `/gateway_chat.html`
- `/gateway_embed.html`

The gateway pages demonstrate browser calls to gateway-profile endpoints. Keep
production server routes in a route-owning framework, an application server, or
the first-party gateway server.

## Related Docs

- [Browser Package](../browser.md)
- [Local Inference](../../guides/local-inference.md)
- [Runtime Options](../../reference/runtime-options.md)
- [Providers](../../guides/providers.md)
- [Gateway Server](../../gateway/server.md)
- [Browser Caching](../../guides/browser-caching.md)
