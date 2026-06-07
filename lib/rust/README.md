# CogentLM Rust Facade

`lib/rust` is the public Rust facade crate published as `cogentlm`. It is the
Rust dependency for local GGUF inference,
gateway-backed inference, provider descriptors, native runtime configuration,
and shared CogentLM value types.

The facade re-exports the high-level `CogentClient` API plus selected engine,
backend, lifecycle, shard, provider, and gateway modules.

## Source Checkout

From the repository root, after `source ./setup.sh`:

```bash
clm build core && cargo run -p cogentlm-rust-examples --bin query -- <model.gguf> "Explain CogentLM."
```

`clm` forwards to `cargo xtask`; use `cargo xtask ...` with the same arguments
if the launcher is not active.

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

Register endpoints with `add(id, descriptor)`, keep the returned `EndpointRef`,
and pass that reference on each request when routing must be explicit.

Gateway clients use `EndpointDescriptor::gateway` when a Rust application calls
a separate CogentLM gateway.

## Learn More

- [Rust package docs](../../docs/packages/rust.md)
- [Local inference](../../docs/guides/local-inference.md)
- [Gateway and hybrid inference](../../docs/guides/gateway-hybrid.md)
- [Rust examples](../../examples/rust/README.md)
