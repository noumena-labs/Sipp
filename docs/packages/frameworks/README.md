# Frameworks

These guides show how to use the JavaScript-facing Sipp packages in common
application frameworks. See the [Library API Overview](../../api) for
the shared `add`, `query`, `chat`, and `embed` contracts.

Use the browser package, `@sipp/sipp`, when inference runs in the browser or when
browser code calls a gateway. Use the Node package, `@sipp/sipp-server`, only in
server-only code such as route handlers, server functions, API routes, workers,
or services that run in a Node.js runtime.

## Guides

- [React And Vite](vite-react.md): Baseline browser-local setup, WebGPU/WASM
  asset behavior, OPFS model loading, and local development headers.
- [Next.js](nextjs.md): App Router provider routes, Client Components,
  gateway-profile compatibility, and streaming.
- [TanStack](tanstack.md): TanStack Start provider functions, server routes,
  and TanStack Query patterns.


## Package Selection

| Environment | Package | Notes |
| --- | --- | --- |
| Browser component | `@sipp/sipp` | Use for browser-local GGUF inference or direct gateway calls. |
| Node server route | `@sipp/sipp-server` | Use for direct provider endpoints, local server inference, or gateway clients. |
| Gateway profile route | `@sipp/sipp-server` | Use when a browser `kind: 'gateway'` endpoint calls a framework route. |
| Gateway client | Either | Browser code can call a separate gateway with short-lived tokens, or server code can use server-held secrets. |

## Provider-First Server Routes

Next.js and TanStack server routes should usually demonstrate direct provider
endpoints when the framework server owns the credential. Register a provider in
server-only code:

```ts
const endpoint = await client.add('provider', {
  kind: 'provider',
  provider: 'openai',
  model: requiredEnv('OPENAI_MODEL'),
  apiKey: requiredEnv('OPENAI_API_KEY'),
});
```

Use `OPENAI_API_KEY="<mock-openai-key>"` only as a placeholder in docs and
examples. Do not expose real provider keys in browser bundles.

## Gateway Route Field Names

Browser gateway descriptors require an absolute `http` or `https` `baseUrl`
and use `routes: { query, chat, embed }` for route overrides. Node gateway
descriptors use `queryRoute`, `chatRoute`, and `embedRoute` when server code
calls a gateway through `@sipp/sipp-server`.

Keep provider credentials and long-lived gateway tokens out of browser bundles.
When a browser app needs gateway access, issue short-lived application tokens or
proxy through a server route.

Use `decodeGatewayQueryBody()`, `decodeGatewayChatBody()`,
`decodeGatewayEmbedBody()`, and the matching response helpers from
`@sipp/sipp-server` when a framework route should be registered as a browser
`kind: 'gateway'` endpoint. Those helpers keep route examples focused on auth,
target policy, provider selection, and client lifecycle instead of gateway
profile JSON shaping.
