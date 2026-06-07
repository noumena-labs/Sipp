# CogentLM Server for Node.js

This package exposes the native `CogentClient` API for Node.js server
processes. It does not provide framework route handlers.

Register local, provider, and gateway endpoints through the same method:

```ts
import { CogentClient } from '@noumena-labs/cogentlm-server'

const client = new CogentClient()

const gateway = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'http://127.0.0.1:8787',
  authentication: {
    kind: 'bearer',
    value: process.env.COGENTLM_GATEWAY_TOKEN,
  },
})

const run = client.chat({
  endpoint: gateway,
  messages: [{ role: 'user', content: 'Explain gateway-backed inference.' }],
})
const response = await run.response
```

Framework integrations should define their own routes and call
`client.query()`, `client.chat()`, or `client.embed()` directly inside those
routes. CogentLM supplies the endpoint client and request shapes; the
application owns request parsing, authentication, routing, and response
encoding.
