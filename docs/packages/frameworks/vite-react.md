# React And Vite

React and Vite are the baseline browser integration for the `cogentlm` package.
The checked-in `examples/web` pages use Vite and demonstrate local query,
chat, embedding, gateway query, gateway chat, gateway embedding, and a
browser-local plus gateway comparison.

## Install

```bash
npm install cogentlm
```

## Browser Local Query

```tsx
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
          runtime: {
            context: { n_ctx: 2048 },
            scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
            cache: { mode: 'live_slot_prefix' },
            observability: { runtime_metrics: true },
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

## Gateway Query

```tsx
import { CogentClient } from 'cogentlm';

const client = new CogentClient();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: {
    kind: 'bearer',
    valueProvider: fetchShortLivedGatewayToken,
  },
});
const run = client.query('Explain gateway inference.', {
  endpoint,
  emitTokens: true,
  maxTokens: 64,
});
```

Gateway clients should use short-lived browser tokens or call through an
application server route. Do not place provider credentials or long-lived
gateway tokens in Vite environment variables that are exposed to browser code.

## WASM Assets

The browser package resolves its packaged Emscripten JavaScript and WASM assets
from the package at runtime. Vite optimized dependency paths are handled by the
package runtime, so most apps can use `new CogentClient()` without asset
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

## Workers And Pthreads

`CogentClient` uses worker execution automatically when the browser environment
supports it. You can force main-thread execution for debugging:

```ts
const client = new CogentClient({ executionMode: 'main-thread' });
```

The pthread runtime requires `SharedArrayBuffer` and cross-origin isolation.
Serve the app with COOP/COEP headers before using:

```ts
const client = new CogentClient({ wasmThreading: 'pthread' });
```

Use `wasmThreading: 'single-thread'` when cross-origin isolation is not
available.

## Model Loading And Cache

Browser local endpoints accept a URL, installed model id, or `File` object as
the `source`. The browser runtime stores model data through OPFS where
available so repeated loads can stay local after the first import or fetch.

Tune browser cache behavior with `browserCache` on the client options and local
runtime cache behavior with `options.runtime.cache` on the endpoint descriptor.

## Existing Examples

Serve the source examples when working from a checkout:

```bash
clm run examples serve browser
```

Then open the printed URL and use:

- `/query.html`
- `/chat.html`
- `/embed.html`
- `/gateway_query.html`
- `/gateway_chat.html`
- `/gateway_embed.html`
- `/gateway_local.html`

## Related Docs

- [Browser Package](../browser.md)
- [Gateway Server](../gateway-server.md)
- [Browser Caching](../../guides/browser-caching.md)
- [Source Builds](../../maintainers/source-builds.md)
