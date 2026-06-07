# CogentLM Rust Facade

## What this library is for

`lib/rust` is the public Rust facade crate published as `cogentlm`. It is the
single Rust dependency applications should use for local GGUF inference,
gateway-backed inference, direct provider adapters, native runtime
configuration, and shared CogentLM value types.

The facade re-exports the high-level `CogentClient` API plus selected engine,
backend, lifecycle, shard, and gateway modules. Register endpoints with
`add(id, descriptor)`, keep the returned `EndpointRef`, and pass that reference
to `query`, `chat`, or `embed` when a request should use a specific endpoint.

## Getting Started

Inside an async function that returns a `Result`:

```rust
use cogentlm::{CogentClient, CogentQueryRequest, CogentTextOptions, EndpointDescriptor, GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes, GatewaySecret, GatewayTimeoutPolicy};
let mut client = CogentClient::new();
let gateway = client.add("gateway", EndpointDescriptor::gateway(GatewayEndpointConfig { target: "local".into(), base_url: "http://127.0.0.1:8787".into(), routes: GatewayRoutes::default(), authentication: GatewayAuthentication::Bearer(GatewaySecret::new(std::env::var("COGENTLM_GATEWAY_TOKEN")?)), static_headers: Default::default(), timeouts: GatewayTimeoutPolicy::default(), protocol_options: Default::default() })).await?;
let response = client.query(CogentQueryRequest { endpoint: Some(gateway), prompt: "Explain gateway inference in one sentence.".into(), options: CogentTextOptions { max_tokens: Some(64), ..Default::default() }, ..Default::default() }).await?;
println!("{}", response.text);
```

The gateway endpoint sends first-party JSON to `base_url` using the default
`/v1/query`, `/v1/chat`, and `/v1/embed` routes. The `target` value is encoded
as the request `model` and resolved by the gateway process.

## Gateway And Hybrid Inference

Hybrid inference is application-owned routing. A single `CogentClient` can hold
a local model endpoint and a gateway endpoint at the same time, then each
request chooses an `EndpointRef`.

```rust
use std::path::PathBuf;

use cogentlm::engine::NativeRuntimeConfig;
use cogentlm::{
    CogentClient, CogentQueryRequest, CogentTextOptions, EndpointDescriptor,
    GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes, GatewaySecret,
    GatewayTimeoutPolicy, LocalTextOptions,
};

async fn run(model_path: PathBuf, prompt: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = CogentClient::new();
    let local = client
        .add(
            "local",
            EndpointDescriptor::local(model_path, NativeRuntimeConfig::default()),
        )
        .await?;
    let gateway = client
        .add(
            "gateway",
            EndpointDescriptor::gateway(GatewayEndpointConfig {
                target: "local".to_string(),
                base_url: "http://127.0.0.1:8787".to_string(),
                routes: GatewayRoutes::default(),
                authentication: GatewayAuthentication::Bearer(GatewaySecret::new(
                    std::env::var("COGENTLM_GATEWAY_TOKEN")?,
                )),
                static_headers: Default::default(),
                timeouts: GatewayTimeoutPolicy::default(),
                protocol_options: Default::default(),
            }),
        )
        .await?;

    let options = CogentTextOptions {
        max_tokens: Some(96),
        temperature: Some(0.7),
        ..Default::default()
    };
    let local_response = client
        .query(CogentQueryRequest {
            endpoint: Some(local),
            prompt: prompt.clone(),
            options: options.clone(),
            local: LocalTextOptions {
                context_key: Some("rust-local".to_string()),
                ..Default::default()
            },
            ..Default::default()
        })
        .await?;
    let gateway_response = client
        .query(CogentQueryRequest {
            endpoint: Some(gateway),
            prompt,
            options,
            ..Default::default()
        })
        .await?;

    println!("local: {}", local_response.text);
    println!("gateway: {}", gateway_response.text);
    Ok(())
}
```

Local-only options such as grammar, media, embedding normalization, and context
keys are valid only for local endpoints. Gateway-specific options go in
`endpoint_options` on each request or `protocol_options` on the gateway
descriptor. Direct provider descriptors are available through
`EndpointDescriptor::provider` when the `providers` feature is enabled, but
secrets are usually better kept in a gateway process.
