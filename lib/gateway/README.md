# cogentlm-gateway

`cogentlm-gateway` is the Rust HTTP toolkit for building application-owned
CogentLM gateway routes. It provides codecs, authentication and observability
traits, HTTP error helpers, and the first-party JSON/SSE profile used by
CogentLM clients.

Applications bind sockets, choose routes, load configuration, define deployment
policy, and call `CogentClient` after decoding a request.

## Source Checkout

From the repository root, after `source ./setup.sh`:

```bash
clm build core && clm run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

`clm` forwards to `cargo xtask`; use `cargo xtask ...` with the same arguments
if the launcher is not active.

## Minimal Handler Shape

```rust
use cogentlm_gateway::{GatewayCodec, ProtocolCodec};

let codec = GatewayCodec;
let mut decoded = codec.decode_query(&body)?;
decoded.request.endpoint = Some(resolve(&decoded.target)?);
let response = client.query(decoded.request).await?;
let bytes = codec.encode_text(&decoded.target, &response)?;
```

`GatewayRoutes::default()` selects `/v1/query`, `/v1/chat`, and `/v1/embed`
for the first-party profile. Applications can choose different routes when
they expose a compatible profile.

Use `apps/gateway-server` for the first-party server application with TOML,
bearer tokens, target policy, CORS, metrics, probes, and deployment behavior.
Node framework routes can use the matching gateway profile helpers exported by
`cogentlm-server`.

## Learn More

- [Gateway toolkit docs](../../docs/gateway/toolkit.md)
- [Gateway architecture](../../docs/gateway/architecture.md)
- [Gateway configuration](../../docs/gateway/configuration.md)
- [Gateway route example](../../examples/gateway/README.md)
