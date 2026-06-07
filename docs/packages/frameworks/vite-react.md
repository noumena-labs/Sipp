# React And Vite

React and Vite are the baseline browser integration for the `cogentlm` package.
This guide focuses on browser-local GGUF inference through the CogentLM
WebAssembly/WebGPU runtime. Vite handles the browser bundle; production
server routes should live in a separate route-owning framework or the
first-party gateway server.

## Install

```bash
npm install cogentlm
```

## Browser Local WebGPU Setup

Use `cogentlm` only in browser code. The local endpoint `source` can be a
model URL served by the app, a user-provided `File`, or an installed model id
returned by `listLocal()`.

When WebGPU is available, request it on the local endpoint load options. The
runtime can fall back to compatible CPU execution when `backend` is omitted.

```ts
const endpoint = await client.add('browser-local', {
  kind: 'local',
  source: '/models/model.gguf',
  options: {
    backend: 'webgpu',
    runtime: {
      context: { n_ctx: 2048 },
      scheduler: { continuous_batching: true, prefill_chunk_size: 0 },
      cache: { mode: 'live_slot_prefix' },
      observability: { runtime_metrics: true },
    },
  },
});
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
          backend: 'webgpu',
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

Keep route setup out of this React/Vite guide for now. When the app needs
server-owned inference, use the first-party gateway server or a framework guide
with production route handlers.

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

For local development, configure Vite dev and preview headers before enabling
the pthread runtime:

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

## Model Loading And Cache

Browser local endpoints accept a URL, installed model id, `File` object,
multiple shard URLs, or multiple shard files as the `source`. The browser
runtime stores model data through OPFS where available so repeated loads can
stay local after the first import or fetch.

```tsx
import { useState } from 'react';
import { CogentClient } from 'cogentlm';

export function FileModelQuery(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState('');

  async function run(): Promise<void> {
    if (file == null) return;
    const client = new CogentClient();
    try {
      const endpoint = await client.add('file-model', {
        kind: 'local',
        source: file,
        options: { backend: 'webgpu' },
      });
      const response = await client.query('Summarize this local model setup.', {
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
      <input
        type="file"
        accept=".gguf"
        onChange={(event) => setFile(event.currentTarget.files?.[0] ?? null)}
      />
      <button type="button" onClick={() => void run()}>
        {text || 'Run'}
      </button>
    </>
  );
}
```

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

The same example app also contains gateway pages, but this guide intentionally
keeps the React/Vite setup focused on browser-local inference.

## Related Docs

- [Browser Package](../browser.md)
- [Gateway Server](../gateway-server.md)
- [Browser Caching](../../guides/browser-caching.md)
- [Source Builds](../../maintainers/source-builds.md)
