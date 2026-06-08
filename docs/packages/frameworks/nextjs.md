# Next.js

Use `cogentlm-server` in App Router route handlers that run in the Node.js
runtime. Use `cogentlm` only in Client Components or browser-only modules.

Next.js App Router pages and layouts are Server Components by default. Add
`'use client'` only to modules that need browser APIs, state, event handlers,
or browser-local CogentLM runtime access.

## Profile-Compatible Server Route

Route handlers are a good place to keep gateway tokens and provider credentials
off the client. Set `runtime = 'nodejs'` for routes that import
`cogentlm-server`.

Routes that are registered from a browser `kind: 'gateway'` endpoint must speak
the first-party gateway profile. Use the gateway profile helpers from
`cogentlm-server` to decode the incoming body and format JSON or SSE responses.

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
    const endpoint = await client.add('gateway', {
      kind: 'gateway',
      target: decoded.target,
      baseUrl: requiredEnv('COGENTLM_GATEWAY_URL'),
      authentication: {
        kind: 'bearer',
        value: requiredEnv('COGENTLM_GATEWAY_TOKEN'),
      },
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

Do not return an app-specific shape such as `{ text }` from a route that the
browser package calls through `client.add({ kind: 'gateway' })`. That route is
an HTTP gateway endpoint from the browser client's perspective, even when it is
implemented inside the Next application.

For high-throughput services, keep endpoint setup in a server-only module and
reuse the client lifecycle according to your deployment model. Do not import
that module from Client Components.

## Streaming Route Handler

Use a route handler when the browser should receive token updates but the
server should keep the gateway token.

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
  const endpoint = await client.add('gateway', {
    kind: 'gateway',
    target: requiredEnv('COGENTLM_GATEWAY_TARGET'),
    baseUrl: requiredEnv('COGENTLM_GATEWAY_URL'),
    authentication: {
      kind: 'bearer',
      value: requiredEnv('COGENTLM_GATEWAY_TOKEN'),
    },
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

## Browser-Local Client Component

Browser-local inference needs browser APIs and should live behind a Client
Component boundary.

```tsx
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

If you override `moduleUrl`, `wasmUrl`, `pthreadModuleUrl`, or
`pthreadWasmUrl`, provide both the JavaScript and WASM asset URLs for the
selected runtime. Use `wasmThreading: 'pthread'` only when the app is served
with cross-origin isolation headers that enable `SharedArrayBuffer`.

## Hybrid Client Component

Use one browser `CogentClient` to register a browser-local endpoint and a
same-origin gateway route. Select the endpoint reference at request time; the
`query` call stays the same.

```tsx
// app/hybrid-chat/HybridChat.tsx
'use client';

import { useState } from 'react';
import { CogentClient, type EndpointRef } from 'cogentlm';

type InferenceMode = 'local' | 'gateway';

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
      const gatewayEndpoint = await client.add('app-route', {
        kind: 'gateway',
        target: 'local',
        baseUrl: window.location.origin,
        routes: { query: '/api/cogent/query' },
        authentication: { kind: 'none' },
      });
      const endpoint: EndpointRef =
        mode === 'local' ? localEndpoint : gatewayEndpoint;
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
        <option value="gateway">Server route</option>
      </select>
      <button type="button" onClick={() => void run('Explain hybrid inference.')}>
        {text || 'Run'}
      </button>
    </>
  );
}
```

Browser gateway descriptors require an absolute `http` or `https` `baseUrl`.
For same-origin Next routes, use `window.location.origin` and set route
overrides such as `routes: { query: '/api/cogent/query' }`.

## Gateway Token Pattern

For direct browser-to-gateway calls, do not embed a long-lived gateway token in
the client bundle. Have a Next route issue a short-lived app token, then use a
browser `valueProvider`:

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

## References

- [Next.js Server and Client Components](https://nextjs.org/docs/app/getting-started/server-and-client-components)
- [Next.js Route Handlers](https://nextjs.org/docs/app/getting-started/route-handlers)
- [Next.js Route Segment Config](https://nextjs.org/docs/app/api-reference/file-conventions/route-segment-config)
