# Rust Package

The Rust package target is `sipp-rs`. It publishes the `sipp` library crate
for Rust applications and re-exports the high-level client API plus selected
runtime, backend, lifecycle, shard, provider, and gateway types.

See the [Library API Overview](../api) for the shared `add`, `query`,
`chat`, and `embed` contracts.

## Install

```bash
cargo add sipp-rs
```

The release workflow publishes `sipp-sys` first, then publishes `sipp-rs`.
Applications depend on the `sipp-rs` package and import the `sipp` crate.

## Use It For

- Rust applications that need local GGUF inference.
- Gateway-backed query, chat, and embedding calls.
- Direct provider descriptors behind the `providers` feature.
- Shared Sipp value types across application boundaries.

## Local GGUF Query

```rust
use sipp::{
    SippClient, SippQueryRequest, SippTextOptions, EndpointDescriptor,
    LocalTextOptions,
};
use sipp::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, KvReuseMode, NativeRuntimeConfig,
    ObservabilityRuntimeConfig, SchedulerRuntimeConfig,
};

async fn run(
    model_path: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = SippClient::new();
    let endpoint = client
        .add(
            "default",
            EndpointDescriptor::local(model_path, runtime_config()),
        )
        .await?;

    let response = client
        .query(SippQueryRequest {
            endpoint: Some(endpoint),
            prompt: "Explain Sipp in one sentence.".to_string(),
            options: SippTextOptions {
                max_tokens: Some(64),
                ..Default::default()
            },
            local: LocalTextOptions {
                context_key: Some("rust-local".to_string()),
                ..Default::default()
            },
            ..Default::default()
        })
        .await?;
    println!("{}", response.text);
    Ok(())
}

fn runtime_config() -> NativeRuntimeConfig {
    NativeRuntimeConfig {
        context: ContextRuntimeConfig {
            n_ctx: Some(2048),
            ..Default::default()
        },
        scheduler: SchedulerRuntimeConfig {
            continuous_batching: true,
            prefill_chunk_size: 0,
            ..Default::default()
        },
        cache: CacheRuntimeConfig {
            mode: KvReuseMode::LiveSlotPrefix,
            ..Default::default()
        },
        observability: ObservabilityRuntimeConfig {
            runtime_metrics: true,
            backend_profiling: false,
        },
        ..Default::default()
    }
}
```

See [Runtime Options](../reference/runtime-options.md) for the shared runtime
config groups and request option boundaries.

## Gateway Query

```rust
use sipp::{
    SippClient, SippQueryRequest, SippTextOptions, EndpointDescriptor,
    GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes, GatewaySecret,
    GatewayTimeoutPolicy,
};

let mut client = SippClient::new();
let endpoint = client
    .add(
        "gateway",
        EndpointDescriptor::gateway(GatewayEndpointConfig {
            target: std::env::var("SIPP_GATEWAY_TARGET")?,
            base_url: std::env::var("SIPP_GATEWAY_URL")?,
            routes: GatewayRoutes::default(),
            authentication: GatewayAuthentication::Bearer(GatewaySecret::new(
                std::env::var("SIPP_GATEWAY_TOKEN")?,
            )),
            static_headers: Default::default(),
            timeouts: GatewayTimeoutPolicy::default(),
            protocol_options: Default::default(),
        }),
    )
    .await?;

let response = client
    .query(SippQueryRequest {
        endpoint: Some(endpoint),
        prompt: "Explain gateway inference.".to_string(),
        options: SippTextOptions {
            max_tokens: Some(64),
            ..Default::default()
        },
        ..Default::default()
    })
    .await?;
println!("{}", response.text);
```

## Related Docs

- [Gateway Server](../gateway/server.md)
- [Gateway Toolkit](../gateway/toolkit.md)
- [Local Inference](../guides/local-inference.md)
- [Providers](../guides/providers.md)
- [Runtime Options](../reference/runtime-options.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Architecture](../architecture.md)
- [Maintainer source builds](../maintainers/source-builds.md)
