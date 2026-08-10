# React And Vite

React and Vite are the baseline browser integration for the `@sipphq/sipp`
package. Use this guide for Vite-specific setup, local development headers,
runtime asset overrides, and the source browser examples.

For the full local inference option map, see
[Local Inference](../../guides/local-inference.md) and
[Runtime Options](../../reference/runtime-options.md).

## Install

```bash
npm install @sipphq/sipp
```

## Browser Local Query

Use `@sipphq/sipp` only in browser code. Add one or more model URLs or
user-provided `File` objects through `client.models`; multiple inputs represent
model shards or a model/projector pair. Pass the `ManagedModel` returned by
`models.add` to `Endpoint.local(...)`.

```ts
import { useState } from 'react';
import { Endpoint, SippClient } from '@sipphq/sipp';

export function LocalQuery(): JSX.Element {
  const [text, setText] = useState('');

  async function run(): Promise<void> {
    const client = new SippClient();
    try {
      const model = await client.models.add([
        '/models/model.gguf',
      ]);
      const endpoint = await client.add(
        'default',
        Endpoint.local(model, {
          backend: 'webgpu',
          runtime: {
            context: { n_ctx: 2048 },
          },
        })
      );
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

Omit `backend` to let the browser runtime choose the backend. Use
`backend: 'webgpu'` when the UI should explicitly request WebGPU and surface
backend errors itself.

## Local Development Headers

The packaged WASM runtime uses pthreads and requires `SharedArrayBuffer` plus
cross-origin isolation. Configure Vite dev and preview headers before using the
default browser runtime:

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

Apps that cannot serve those headers must provide custom single-thread
assets with `wasmThreading: 'single-thread'`, `moduleUrl`, and `wasmUrl`.

## Runtime Asset Overrides

The browser package resolves its packaged Emscripten JavaScript and WASM assets
at runtime. Most Vite apps can use `new SippClient()` without asset
overrides.

Override runtime asset URLs only when your bundler or deployment moves package
assets:

```ts
const client = new SippClient({
  moduleUrl: '/assets/sipp-wasm-pthread.js',
  wasmUrl: '/assets/sipp-wasm-pthread.wasm',
});
```

`moduleUrl` and `wasmUrl` override the selected runtime. The selected runtime
defaults to pthread. Custom single-thread builds must also set
`wasmThreading: 'single-thread'`.

## Model Files And Cache

Serve model URLs from the application or let users select local `.gguf` files.
The browser runtime stores model data through OPFS where available, so repeated
loads can stay local after the first import or fetch.

Tune browser storage with `browserCache` on `SippClient` and tune local
runtime behavior with `options.runtime` on the local endpoint. See
[Browser Caching](../../guides/browser-caching.md) and
[Runtime Options](../../reference/runtime-options.md).

## Existing Examples

Serve the source examples when working from a checkout:

```bash
sipp run examples serve browser
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
