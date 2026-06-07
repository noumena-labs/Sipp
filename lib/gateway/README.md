# cogentlm-gateway

## What this library is for

`cogentlm-gateway` is the Rust HTTP toolkit for building CogentLM gateway
routes without adopting a full server application. It provides codecs,
authentication and observability traits, HTTP error helpers, and the
first-party JSON/SSE profile used by CogentLM clients.

This crate does not bind sockets, choose routes, load configuration, or define
deployment policy. Your application owns those boundaries and calls
`CogentClient` after decoding a request.

## Getting Started

Inside a route handler with a configured `CogentClient`, request `body`, and a
`resolve(target)` function:

```rust
use cogentlm_gateway::{GatewayCodec, ProtocolCodec};
let codec = GatewayCodec;
let mut decoded = codec.decode_query(&body)?;
decoded.request.endpoint = Some(resolve(&decoded.target)?);
let bytes = codec.encode_text(&decoded.target, &client.query(decoded.request).await?)?;
```

`GatewayCodec` understands the first-party profile. Query and chat requests use
the target name in `model`; embedding requests use `model` and `input`.

## Endpoint Routes

`GatewayRoutes::default()` selects these HTTP paths for the first-party
profile:

* `/v1/query`
* `/v1/chat`
* `/v1/embed`

The optional index, health, readiness, and metrics paths are also carried in
`GatewayRoutes`, but this toolkit does not implement handlers for them. Use
`apps/gateway-server` if you want the opinionated first-party application.

## Gateway And Hybrid Inference

Gateway handlers usually have three application-owned pieces:

* a `CogentClient` with local and/or provider endpoints already registered
* a target map from public target names to `EndpointRef` values
* a codec that translates HTTP bodies to typed CogentLM requests

```rust
use axum::{body::Body, extract::State, http::{Response, StatusCode}};
use bytes::Bytes;
use cogentlm::{CogentClient, EndpointRef};
use cogentlm_gateway::{GatewayCodec, GatewayHttpError, ProtocolCodec};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone)]
struct AppState {
    client: Arc<CogentClient>,
    targets: Arc<BTreeMap<String, EndpointRef>>,
    codec: GatewayCodec,
}

async fn query(State(state): State<AppState>, body: Bytes) -> Response<Body> {
    let decoded = match state.codec.decode_query(&body) {
        Ok(decoded) => decoded,
        Err(error) => return error_response(&state, error),
    };
    let Some(endpoint) = state.targets.get(&decoded.target).cloned() else {
        return error_response(
            &state,
            GatewayHttpError::new(StatusCode::NOT_FOUND, "resolution", "target not found"),
        );
    };

    let mut request = decoded.request;
    request.endpoint = Some(endpoint);
    request.emit_tokens = decoded.stream;

    match state.client.query(request).await {
        Ok(response) => match state.codec.encode_text(&decoded.target, &response) {
            Ok(body) => Response::builder()
                .header("content-type", state.codec.content_type(false))
                .body(Body::from(body))
                .unwrap_or_else(|_| Response::new(Body::empty())),
            Err(error) => error_response(&state, error),
        },
        Err(error) => error_response(&state, GatewayHttpError::from_gateway_error(error.into())),
    }
}
```

That same target map can point some public names at local GGUF endpoints and
others at direct provider endpoints. The gateway route stays the same because
`decoded.target` is just the application target name, and the resolved
`EndpointRef` controls where inference runs.

For streaming query and chat responses, set `request.emit_tokens = true`, read
the returned token batches, and encode `GatewayStreamEvent` values with
`GatewayCodec::encode_stream_event`. The minimal runnable Axum version is in
`examples/gateway`; production auth, CORS, metrics, probes, and TOML live in
`apps/gateway-server`.

## Protocol Shape

The first-party profile accepts these typed JSON bodies:

* query: `model`, `prompt`, optional text options, and `stream`
* chat: `model`, `messages`, optional text options, and `stream`
* embed: `model`, `input`, and endpoint-specific options

Unknown profile fields are carried as endpoint options. Client libraries expose
those as `endpointOptions` or `protocolOptions`, so application-specific
gateway flags can pass through without changing the core client API.
