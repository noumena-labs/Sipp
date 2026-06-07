# CogentLM Gateway Primitives

`cogentlm-gateway` contains framework-neutral building blocks for exposing a
configured `cogentlm-client` through an HTTP or application framework.

It owns:

- protocol request and response envelopes
- public alias to private endpoint routing
- caller access scopes
- replica-local concurrency, rate, and quota limits
- canonical request context and cancellation reasons
- unary and streaming execution over `GatewayExecutor`
- typed gateway errors, including `server_restarting`

It does not own listeners, TLS, bearer parsing, CORS, configuration files,
logging, metrics, dashboards, or deployment.

## Minimal Construction

```rust
let mut client = cogentlm_client::CogentClient::new();
let endpoint = client
    .add(
        "private-local",
        cogentlm_client::EndpointDescriptor::local(
            "model.gguf",
            cogentlm_engine::engine::NativeRuntimeConfig::default(),
        ),
    )
    .await?;

let gateway = cogentlm_gateway::GatewayAdapter::builder(
    cogentlm_gateway::CogentClientExecutor::new(client),
)
.alias(cogentlm_gateway::GatewayAlias::new(
    "public-model",
    endpoint,
    cogentlm_gateway::OperationSet::all(),
    cogentlm_gateway::GatewayAliasLimits::default(),
)?)?
.build()?;
```

Every execution receives a `GatewayRequestContext`. Framework adapters should
cancel its `GatewayCancellation` when the downstream client disconnects.
Dropping or cancelling the resulting run aborts local engine work or upstream
HTTP work and releases concurrency permits.

See `examples/gateway` for a small Axum adapter and `apps/gateway-server` for
the production service.
