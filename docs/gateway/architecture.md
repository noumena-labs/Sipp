# Gateway Architecture

Gateway behavior is split into independent layers. There is no compatibility
layer for deleted gateway route-autowiring or remote endpoint APIs.

## Core Execution

`sipp::gateway_core` (the `gateway_core` module of the `sipp` crate,
behind the `gateway` feature) exposes only typed query, chat, and embed
execution:

- `GatewayRequestContext` and cancellation.
- `TargetResolver`, `Authorizer`, `AdmissionController`, and
  `GatewayExecutor`.
- `GatewayPipeline` ordering and admission-permit lifetime.
- Protocol-neutral finite results and streaming events.

It does not depend on HTTP, Axum routes, JSON, SSE, bearer tokens, status
codes, aliases, TOML, or fixed limits.

The `sipp` client API owns local, provider, and gateway endpoint
registration through `SippClient.add(...)`. Gateway endpoints call an HTTP
gateway as a client transport and are never selected implicitly.

## Developer Toolkit

`lib/gateway` contains route-free HTTP helpers for applications that choose to
expose a gateway:

- `ProtocolCodec` for request, response, stream, and error wire formats.
- `Authenticator` for arbitrary authentication.
- `ErrorTranslator` for application HTTP error mapping.
- `GatewayCodec` for the first-party Sipp JSON/SSE profile.
- `GatewayHttpError` and SSE/error response encoders.

It does not register routes, expose a router, or own handler paths.
Applications decode requests, select targets, call `client.query()`,
`client.chat()`, or `client.embed()` directly, and encode responses
explicitly.

## Public Endpoints

Rust, Node, Python, and browser packages expose gateway endpoint descriptors
through the same `.add` path used for local and provider endpoints:

- A protocol target.
- A gateway base URL.
- Query, chat, and embed routes.
- Authentication strategy.
- Static headers.
- Timeout policy.
- Protocol-specific request options.

The endpoint id is supplied only to `.add`. Local model, provider, and gateway
descriptors are different descriptor kinds, but `query`, `chat`, and `embed`
request shapes are identical once an endpoint ref is selected.

## First-Party Applications

`apps/gateway-server` is one opinionated first-party application. Its bearer
tokens, target access, concurrency limit, CORS, routes, management listener,
metrics, and TOML format are application-owned.

`examples/gateway` demonstrates the canonical developer pattern:

- Create a `SippClient`.
- Add local, provider, or gateway endpoints with `.add`.
- Define Axum routes in the example application.
- Decode each route body, select an endpoint, call `client.*`, and encode the
  response.

Default `/v1/query`, `/v1/chat`, and `/v1/embed` paths belong only to
applications that choose them. The library supplies codecs and endpoint
transports, not route ownership.
