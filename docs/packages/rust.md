# Rust Package

The Rust package target is `cogentlm`. It is the public facade crate for Rust
applications and re-exports the high-level client API plus selected runtime,
backend, lifecycle, shard, provider, and gateway types.

## Use It For

- Rust applications that need local GGUF inference.
- Gateway-backed query, chat, and embedding calls.
- Direct provider descriptors behind the `providers` feature.
- Shared CogentLM value types across application boundaries.

## Local GGUF Query

```rust
use cogentlm::{
    CogentClient, CogentQueryRequest, CogentTextOptions, EndpointDescriptor,
    LocalTextOptions,
};
use cogentlm::engine::{
    CacheRuntimeConfig, ContextRuntimeConfig, KvReuseMode, NativeRuntimeConfig,
    ObservabilityRuntimeConfig, SchedulerRuntimeConfig,
};

async fn run(
    model_path: std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = CogentClient::new();
    let endpoint = client
        .add(
            "default",
            EndpointDescriptor::local(model_path, runtime_config()),
        )
        .await?;

    let response = client
        .query(CogentQueryRequest {
            endpoint: Some(endpoint),
            prompt: "Explain CogentLM in one sentence.".to_string(),
            options: CogentTextOptions {
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

## Gateway

Register `EndpointDescriptor::gateway` when a Rust application calls a separate
CogentLM gateway. The gateway toolkit and server docs cover route and
deployment ownership.

## Related Docs

- [Local Inference](../guides/local-inference.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Architecture](../architecture.md)
