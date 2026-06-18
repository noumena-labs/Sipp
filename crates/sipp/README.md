# Sipp Rust Library

`crates/sipp` is the public Rust facade crate published as the `sipp-rs`
package. Applications depend on `sipp-rs` and import the `sipp` crate for
local GGUF inference, gateway-backed inference, provider descriptors, native
runtime configuration, and shared Sipp value types.

`sipp-rs` depends on `sipp-sys`, the native llama.cpp FFI crate. A downstream
`cargo add sipp-rs` build therefore needs Rust, a C/C++ compiler, CMake, and a
CMake generator such as Ninja. Optional backend features require their platform
SDKs: CUDA Toolkit for `cuda`, Xcode command line tools on macOS for `metal`,
Vulkan development libraries for `vulkan`, and OpenMP support for `openmp`.

The crate exposes the high-level `SippClient` API at the root plus the
`engine`, `backend`, `lifecycle`, `runtime`, `core`, `shard`, `error`,
`providers` (feature `providers`), and `gateway_core` (feature `gateway`)
modules.

## Source Checkout

From the repository root, after `source ./setup.sh`:

```bash
sipp build core && cargo run -p sipp-rust-examples --bin query -- <model.gguf> "Explain Sipp."
```

`sipp` forwards to `cargo xtask`; use `cargo xtask ...` with the same arguments
if the launcher is not active.

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

Register endpoints with `add(id, descriptor)`, keep the returned `EndpointRef`,
and pass that reference on each request when routing must be explicit.

Gateway clients use `EndpointDescriptor::gateway` when a Rust application calls
a separate Sipp gateway.

## Learn More

- [Rust package docs](../../docs/en/packages/rust.md)
- [Local inference](../../docs/en/guides/local-inference.md)
- [Gateway and hybrid inference](../../docs/en/guides/gateway-hybrid.md)
- [Rust examples](../../examples/rust/README.md)
