# TanStack

TanStack apps usually need two CogentLM patterns:

- TanStack Start server functions for server-only CogentLM work, gateway
  tokens, provider credentials, and local model paths.
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
browser-local GGUF inference and direct browser gateway endpoints with
short-lived tokens.

```ts
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
```

Do not import `cogentlm-server` from browser modules.

## References

- [TanStack Start Server Functions](https://tanstack.com/start/latest/docs/framework/react/guide/server-functions)
- [TanStack Query useQuery](https://tanstack.com/query/latest/docs/framework/react/reference/useQuery)
