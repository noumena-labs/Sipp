# Gateway Toolkit

`cogentlm-gateway` is a route-free Rust HTTP toolkit for applications that want
to expose CogentLM inference through their own server framework.

The toolkit provides codecs, authentication and observability traits, HTTP
error helpers, and the first-party JSON/SSE profile. Applications bind sockets,
register routes, load configuration, and define deployment policy.

Use [Gateway Server](server.md) when you want the first-party server
application with TOML, bearer tokens, target policy, metrics, probes, and
listener management.

## Distribution

The toolkit crate target is `cogentlm-gateway`. crates.io publishing covers
the `cogentlm` and `cogentlm-sys` crates; the toolkit is intentionally
source-distributed. Use [Source Builds](../maintainers/source-builds.md) when
consuming the toolkit from this checkout.

## Use It For

- Building application-owned HTTP gateway routes.
- Translating request bodies into typed CogentLM requests.
- Encoding JSON and SSE responses.
- Sharing the first-party protocol profile with CogentLM clients.

## Minimal Handler Shape

```rust
use cogentlm_gateway::{GatewayCodec, ProtocolCodec};

let codec = GatewayCodec;
let mut decoded = codec.decode_query(&body)?;
decoded.request.endpoint = Some(resolve(&decoded.target)?);
let response = client.query(decoded.request).await?;
let bytes = codec.encode_text(&decoded.target, &response)?;
```

Custom gateway applications own sockets, route layout, authentication,
configuration files, target policy, CORS, logging, and deployment defaults.
Node route handlers can use the matching gateway profile helpers exported by
`cogentlm-server` when implementing the same first-party profile in framework
routes.

## Boundaries

`lib/gateway` supplies helpers, not an application:

- It does not register routes.
- It does not bind listeners.
- It does not own bearer-token policy.
- It does not own TOML, CORS, metrics, or deployment behavior.

Default `/v1/query`, `/v1/chat`, and `/v1/embed` paths belong only to
applications that choose them.

## Related Docs

- [Architecture](architecture.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Frameworks](../packages/frameworks/README.md)
- [Source Builds](../maintainers/source-builds.md)

