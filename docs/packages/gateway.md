# Gateway Toolkit

`cogentlm-gateway` is a route-free Rust HTTP toolkit for applications that want
to expose CogentLM inference through their own server framework.

The toolkit provides codecs, authentication and observability traits, HTTP
error helpers, and the first-party JSON/SSE profile. Applications bind sockets,
register routes, load configuration, and define deployment policy.

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

Use `apps/gateway-server` when you want the first-party server application with
TOML, bearer tokens, target policy, metrics, probes, and listener management.

## Related Docs

- [Gateway Architecture](../gateway.md)
- [Gateway Server](../reference/gateway-server.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
