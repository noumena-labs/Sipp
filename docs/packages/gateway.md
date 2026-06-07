# Gateway Toolkit

`cogentlm-gateway` is a route-free Rust HTTP toolkit for applications that want
to expose CogentLM inference through their own server framework.

The toolkit provides codecs, authentication and observability traits, HTTP
error helpers, and the first-party JSON/SSE profile. Applications bind sockets,
register routes, load configuration, and define deployment policy.

Use [Gateway Server](gateway-server.md) when you want the first-party server
application with TOML, bearer tokens, target policy, metrics, probes, and
listener management.

## Distribution

The toolkit crate target is `cogentlm-gateway`. The current release workflow
packages Rust source artifacts but does not publish Rust crates to crates.io.
Use [Source Builds](../maintainers/source-builds.md) when consuming the toolkit
from this checkout until Rust crate publishing is enabled.

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

## Related Docs

- [Gateway Server](gateway-server.md)
- [Gateway Architecture](../gateway.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Maintainer source builds](../maintainers/source-builds.md)
