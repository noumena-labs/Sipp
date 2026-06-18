# Rust 包

Rust 包发布名称为 `sipp-rs`，导入的库 crate 名仍然是 `sipp`。它作为 Rust 应用的公共 Crate，负责客户端 API 以及运行时、后端、生命周期、分片、提供商和网关等核心类型。

`sipp-rs` 依赖 `sipp-sys`，也就是 llama.cpp 原生 FFI crate。从 crates.io
安装 `sipp-rs` 时，会在目标机器上从源码编译原生后端；它不是类似二进制
Wheel 的预编译包。

各平台共享的 `add`、`query`、`chat`、`embed` 见[API 概述](../api)。

## 安装

```bash
cargo add sipp-rs
```

发布流程会先发布 `sipp-sys`，再发布 `sipp-rs`。应用依赖 `sipp-rs` 包，并在代码中导入 `sipp` crate。需直接从源码使用该包时，见[源码构建](../maintainers/source-builds.md)。

## 构建要求

依赖 `sipp-rs` 的 Rust 应用需要常规 Rust 工具链，以及 `sipp-sys` 使用的原生
构建工具：

- 目标平台可用的 C/C++ 编译器。
- CMake。
- Ninja，或兼容的 CMake generator。
- 所选后端需要的平台 SDK。

默认构建包含 CPU 后端。其他后端特性有额外要求：

- `cuda`：CUDA Toolkit 以及兼容的 NVIDIA 驱动。
- `metal`：macOS 以及 Xcode command line tools。
- `vulkan`：Vulkan SDK 或系统 Vulkan 开发库。
- `openmp`：目标平台上的 OpenMP 编译器/运行时支持。

## 适用场景

- Rust 应用中执行本地 GGUF 推理。
- 通过网关发起 query、chat、embedding 调用。
- 启用 `providers` 特性后，直接使用提供商描述符调用外部 API。
- 在应用的不同模块间共享 Sipp 数据类型。

## 本地推理 (Query)

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

关于共享的运行时配置组与请求选项的说明，请参阅[运行时选项](../reference/runtime-options.md)。

## 网关推理

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

## 相关文档

- [网关服务器](../gateway/server.md)
- [网关工具包](../gateway/toolkit.md)
- [本地推理](../guides/local-inference.md)
- [提供商](../guides/providers.md)
- [运行时选项](../reference/runtime-options.md)
- [网关与混合推理](../guides/gateway-hybrid.md)
- [架构](../architecture.md)
- [维护者源码构建](../maintainers/source-builds.md)
