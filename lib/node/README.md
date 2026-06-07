# CogentLM Server for Node.js

The package exposes the native `CogentClient` API and a Next.js App Router
gateway adapter.

## Next.js App Router

```ts
// app/api/cogentlm/[operation]/route.ts
import { createNextGateway } from '@noumena-labs/cogentlm-server/next'

export const runtime = 'nodejs'

const handler = createNextGateway({
  aliases: {
    local: {
      kind: 'local',
      modelPath: process.env.COGENTLM_MODEL_PATH!,
    },
  },
  auth: async (request) => {
    return request.headers.get('authorization') === `Bearer ${process.env.COGENTLM_TOKEN}`
  },
  maxRequestBytes: 1 << 20,
})

export const POST = handler
```

For local development, authentication must still be explicit:

```ts
const handler = createNextGateway({
  aliases,
  auth: 'none',
})
```

Configure the native package as external:

```js
// next.config.js
module.exports = {
  serverExternalPackages: ['@noumena-labs/cogentlm-server'],
}
```

The adapter supports finite JSON responses, SSE, request body limits,
`x-request-id`, typed gateway errors, and `Request.signal` cancellation. It
supports App Router on the Node.js runtime only; Edge runtime and Pages Router
are intentionally unsupported.
