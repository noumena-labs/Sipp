# TanStack

TanStack apps usually need two CogentLM patterns:

- TanStack Start server functions for server-only CogentLM work, gateway
  tokens, provider credentials, local model paths, and typed app RPC.
- TanStack Start server routes when browser code should register the route as
  a `kind: 'gateway'` endpoint through the CogentLM browser package.
- TanStack Query for client-side final responses that can be cached or
  refetched by query key.

Use explicit component state or a custom hook for token streaming. TanStack
Query is best for Promise-shaped final data, not for appending token batches as
they arrive.

## TanStack Start Server Function

Server functions run on the server and can be called from loaders, components,
hooks, or other server functions. Keep `cogentlm-server`, provider credentials,
and gateway tokens in server-only functions.

```ts
// src/server/cogent.ts
import { createServerFn } from '@tanstack/react-start';
import { CogentClient } from 'cogentlm-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

export const queryCogent = createServerFn({ method: 'POST' })
  .inputValidator((data: { prompt: string }) => data)
  .handler(async ({ data }) => {
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
      prompt: data.prompt,
      options: { maxTokens: 128 },
    });
    const response = await run.response;
    return { text: response.text, usage: response.usage };
  });
```

Validate server-function inputs with the same rigor as any public endpoint.
Server functions are callable network endpoints, so apply application auth and
tenant checks inside the function or middleware.

Server functions are a good fit for typed application calls that return
application-owned shapes such as `{ text }`. They are not the right surface for
browser `client.add({ kind: 'gateway' })` endpoints, because those endpoints
expect the first-party gateway HTTP profile.

## TanStack Start Gateway Route

Use a server route when the browser package should call the framework route as
a gateway endpoint. The route accepts the first-party query profile and returns
the fields consumed by browser gateway endpoints. The gateway profile helpers
decode the browser request and format JSON or SSE responses.

```ts
// src/routes/api/cogent/query.ts
import { createFileRoute } from '@tanstack/react-router';
import {
  CogentClient,
  decodeGatewayQueryBody,
  gatewayErrorResponse,
  gatewayTextResponseBody,
  gatewayTextStreamResponse,
} from 'cogentlm-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

export const Route = createFileRoute('/api/cogent/query')({
  server: {
    handlers: {
      POST: async ({ request }) => {
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
      },
    },
  },
});
```

This route uses the browser profile field `model` as the public gateway target
and keeps the long-lived gateway token on the server. Add application auth or
target allowlists before exposing the route to users.

## TanStack Query For Final Responses

Use TanStack Query when the UI needs a final response and normal query cache
behavior.

```tsx
import { useQuery } from '@tanstack/react-query';
import { queryCogent } from '../server/cogent';

export function Answer({ prompt }: { readonly prompt: string }): JSX.Element {
  const result = useQuery({
    queryKey: ['cogent-query', prompt],
    queryFn: () => queryCogent({ data: { prompt } }),
    enabled: prompt.trim() !== '',
  });

  if (result.isPending) return <p>Loading...</p>;
  if (result.isError) return <p>{result.error.message}</p>;
  return <pre>{result.data.text}</pre>;
}
```

Keep the query key tied to the prompt, target, and any user-visible generation
options that change the result.

## Streaming Tokens

For token streaming, create a server route or server function that returns a
stream, then append chunks with component state.

```tsx
import { useState } from 'react';

export function StreamingAnswer(): JSX.Element {
  const [text, setText] = useState('');

  async function run(prompt: string): Promise<void> {
    setText('');
    const response = await fetch('/api/cogent/stream', {
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

## Browser Package

Use browser `cogentlm` from components that run in the browser. That includes
browser-local GGUF inference and gateway endpoints with short-lived tokens or
same-origin server routes.

```tsx
import { useState } from 'react';
import { CogentClient } from 'cogentlm';

export function LocalAnswer(): JSX.Element {
  const [text, setText] = useState('');

  async function run(prompt: string): Promise<void> {
    const client = new CogentClient();
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

Do not import `cogentlm-server` from browser modules.

## Browser Hybrid Endpoints

Register browser-local and same-origin gateway endpoints on one browser
`CogentClient`, then choose the endpoint reference for each request.

```tsx
import { useState } from 'react';
import { CogentClient, type EndpointRef } from 'cogentlm';

type InferenceMode = 'local' | 'gateway';

export function HybridAnswer(): JSX.Element {
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

Browser gateway descriptors need an absolute `http` or `https` `baseUrl`.
Same-origin TanStack routes should use `window.location.origin` and route
overrides such as `routes: { query: '/api/cogent/query' }`.

## References

- [TanStack Start Server Functions](https://tanstack.com/start/latest/docs/framework/react/guide/server-functions)
- [TanStack Start Server Routes](https://tanstack.com/start/latest/docs/framework/react/guide/server-routes)
- [TanStack Query useQuery](https://tanstack.com/query/latest/docs/framework/react/reference/useQuery)
