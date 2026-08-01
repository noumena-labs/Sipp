
<p align="center">
  <img src="docs/assets/sipp_logo_no_text.svg" alt="Sipp Logo" width="200">
</p>

<div id="user-content-toc" align="center">
  <ul style="list-style: none;">
    <summary>
      <h1>Sipp</h1>
    </summary>
  </ul>
</div>

<p align="center">
  <strong>Serious AI infrastructure. Packaged simply.</strong>
</p>

---


<p align="center">
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/noumena-labs/Sipp/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/noumena-labs/Sipp/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/coverage.yml"><img alt="Coverage" src="https://github.com/noumena-labs/Sipp/actions/workflows/coverage.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/release.yml"><img alt="Release" src="https://github.com/noumena-labs/Sipp/actions/workflows/release.yml/badge.svg"></a>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue">
</p>

<div align="center">
  <a href="docs/en/getting-started/quickstarts.md">Documentation</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://discord.gg/abzgfghhrq">Discord</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://github.com/noumena-labs/Sipp/issues">Issues</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="docs/en/roadmap.md">Roadmap</a>
  <br />
  <a href="README_zh.md">中文介绍</a>
  <br />
</div>



> [!WARNING]
> Sipp is under active development. Breaking changes are expected as we optimize the runtime layers. It might not be suitable for mission-critical production environments yet. If you find issues, bugs, or missing features, please open a GitHub issue.

### [Read the documentation →](docs/en/README.md)
### [中文文档 →](docs/zh/README.md)

## What is Sipp?

Sipp is an all-in-one, high-performance AI framework for building web, desktop, and edge applications. It ships as a cohesive SDK with a unified, symmetric API for local, provider, and cloud gateway inference.

At its core is **Sipp Engine**, a blazing-fast runtime built to run anywhere: in the browser, on the desktop, or on bare-metal cloud infrastructure, that delivers low startup times and a minimal memory footprint.

```javascript
import { EndpointDescriptor, SippClient } from '@sipphq/sipp';

const blender = new SippClient();

// 1. Register a model and load it into a local WebGPU endpoint.
const edgeModel = await blender.models.add(['/models/llama3.gguf']);
const juice = await blender.add(
  'edge',
  EndpointDescriptor.local(edgeModel.id, { backend: 'webgpu' })
);

// 2. Or connect to a secure cloud gateway through the same client.
const ice = await blender.add(
  'cloud',
  EndpointDescriptor.gateway({
    target: 'production',
    baseUrl: 'https://gateway.example.com',
    authentication: { kind: 'bearer', value: '<short-lived-token>' },
  })
);

// Route the same operation to either endpoint.
const [smoothie, snowcone] = await Promise.all([
  blender.chat([{ role: 'user', content: 'Explain Sipp.' }], { endpoint: juice }).response,
  blender.chat([{ role: 'user', content: 'Create a Sipp app.' }], { endpoint: ice }).response,
]);

console.log(smoothie.text, snowcone.text);
await blender.close();
```

The unified SDK lets you dynamically partition and optimize complex application logic between local and cloud compute. Instead of wrestling with fragmented web runtimes, disconnected native wrappers for desktop, or custom middleware to protect API keys, you only need Sipp.

It packages a **high-performance WebGPU engine**, with a secure container gateway proxy into a single, neat toolkit. Future releases will focus on embedded vector memory, on-device PII masking, and automated smart routing. See [Roadmap](docs/en/roadmap.md).

From a source checkout, use the repo launcher installed by the setup scripts,
or call the underlying xtask directly:

```bash
sipp build wasm                  # Compile high-performance WebGPU assets
sipp run demos serve chat        # Launch a hardware-accelerated demo

# Equivalent commands without the repo launcher:
cargo xtask build wasm
cargo xtask run demos serve chat
```


## Performance Benchmarks

Run them yourself here: [benchmark.sipp.sh/benchmark](https://benchmark.sipp.sh/benchmark)

| Runtime / Framework | TTFT (ms) ↓ | Decode (tok/s) ↑ | E2E Latency (ms) ↓ |
| --- | --- | --- | --- |
| **Sipp** | **24.3** *(Best)* | **77.07** *(Best)* | **6,655** *(Best)* |
| **WebLLM** | 160.0 *(6.55x)* | 25.80 *(2.99x)* | 19,930 *(2.99x)* |
| **Transformers.js** | 301.0 *(12.38x)* | 33.25 *(2.32x)* | 15,670 *(2.35x)* |

---

> **Disclaimer & Metric Notes:**
> * **TTFT (Time to First Token):** Measured in milliseconds (ms). **Lower is better**.
> * **Decode:** Measured in tokens per second (tok/s). **Higher is better**.
> * **E2E Latency (End-to-End Latency):** Measured in milliseconds (ms). **Lower is better**.
> * *Performed on a Nvidia GTX 3080, 1 warm up, 3 measured runs. Results avg. of all measured runs.*



## Install

Sipp supports web browsers, desktop application wrappers, server environments, and native runtimes. Install the specific implementation layer for your surface environment:

```sh
# For Web Browsers, Next.js, and TanStack applications
npm install @sipphq/sipp

# For Node.js backend deployments (with native CUDA/Metal compilation)
npm install @sipphq/sipp-server

# For native systems development and application embedding
cargo add sipp-rs

# For native macOS and iOS apps (source checkout on macOS)
sipp build swift

# For Python automation and data engineering pipelines
# (sippy wheels ship from GitHub Releases today; full PyPI build matrix in progress)
# pip install sipppy

# Deploy the secure cloud gateway server instance via Docker
# (cloud gateway will be available in the future, currently building from source)
# docker pull noumena/sipp-gateway
```

---

## Runtimes & Flavors

Most developers should start with our pre-built packages when available. Swift
support currently builds from a source checkout on macOS.

| Surface | Module | Install | Docs |
| --- | --- | --- | --- |
| **Browser** | Sipp Edge | `npm install @sipphq/sipp` | [Browser package](docs/en/packages/browser.md) |
| **Node.js** | Sipp Core | `npm install @sipphq/sipp-server` | [Node.js package](docs/en/packages/node.md) |
| **Rust** | Sipp Core | `cargo add sipp-rs` | [Rust package](docs/en/packages/rust.md) |
| **Python** | Sipp Core | Wheels available on release page | [Python package](docs/en/packages/python.md) |
| **Swift** | Sipp Core | Source-built on macOS | [Swift package](docs/en/packages/swift.md) |
| **Gateway Server** | Sipp Cloud | Source-built | [Gateway Server](docs/en/gateway/server.md) |
| **Gateway Toolkit** | Sipp Cloud | Source-built | [Gateway toolkit](docs/en/gateway/toolkit.md) |

---

## Quick Starts

### 1. Edge Quick Start (Hardware-Accelerated Client Inference)

Initialize the local engine client to execute model weights directly on the client machine's shader cores using WebGPU.

```bash
npm install @sipphq/sipp
```

```javascript
import { EndpointDescriptor, SippClient } from '@sipphq/sipp';

const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain Sipp in one sentence.' },
];

const client = new SippClient();
const model = await client.models.add(['/models/model.gguf']);
const endpoint = await client.add(
  'default',
  EndpointDescriptor.local(model.id, {
    backend: 'webgpu',
    runtime: { context: { n_ctx: 2048 } },
  })
);

const run = client.chat(messages, {
  endpoint,
  maxTokens: 64,
});

console.log((await run.response).text);
await client.close();
```

### 2. Cloud Gateway Quick Start (Preemptive Cloud Proxying)

Cloud gateway clients use the same `SippClient` API layout. The gateway owns
model paths, provider credentials, access policies, and centralized metrics;
your client only needs its public target, URL, and authentication.

```javascript
import { EndpointDescriptor, SippClient } from '@sipphq/sipp';

const client = new SippClient();
const endpoint = await client.add(
  'gateway',
  EndpointDescriptor.gateway({
    target: 'upstream-cluster',
    baseUrl: 'https://gateway.example.com',
    authentication: { kind: 'bearer', value: await getGatewayToken() },
  })
);

const run = client.query('Explain gateway inference.', {
  endpoint,
  maxTokens: 64,
});

console.log((await run.response).text);
await client.close();
```

### 3. Swift Quick Start (Native macOS and iOS)

Swift support requires macOS and Xcode. The build stages a local Swift package
at `.build/artifacts/swift/package` for macOS and iOS.

```bash
sipp toolchain install rust-apple
sipp doctor --target swift
sipp build swift
```

```swift
import Foundation
import Sipp

let modelURL = URL(fileURLWithPath: "/path/to/model.gguf")
let client = try SippClient()
let model = try await client.models.add([modelURL])
try await client.add("local", model: model)

let run = client.chat(
    messages: [
        ChatMessage(role: .system, content: "Answer concisely."),
        ChatMessage(role: .user, content: "Explain Sipp in one sentence."),
    ],
    endpoint: "local",
    options: TextOptions(maxTokens: 64)
)

for await batch in run.tokens {
    print(batch.text, terminator: "")
}

let response = try await run.response
print(response.text)
```

Run the staged command-line example, open the sandboxed macOS app, or launch
the iOS example in a simulator. Build the optional example artifacts first:

```bash
sipp build swift --examples

.build/artifacts/swift/examples/SippCLI chat \
  /path/to/model.gguf "Explain on-device inference"

open .build/artifacts/swift/examples/SippSandbox.app

xcrun simctl list devices available
sipp run examples ios --simulator <simulator-udid> \
  --model /path/to/model.gguf
```

See the [Swift package guide](docs/en/packages/swift.md) and
[Swift examples](examples/swift/README.md) for sandbox entitlements, supported
operations, and device builds.

---

## Native Web Framework Blueprints

Sipp includes native integration blueprints to handle Server-Sent Events (SSE) streaming, serverless route orchestration, and client hydration patterns out of the box.

- [Next.js](docs/en/packages/frameworks/nextjs.md): App Router route handlers,
  Client Components, gateway proxies, and streaming.
- [TanStack](docs/en/packages/frameworks/tanstack.md): TanStack Start server
  functions and TanStack Query patterns.
- [React And Vite](docs/en/packages/frameworks/vite-react.md): Browser package
  setup, WASM assets, OPFS model loading, and gateway examples.

  
## Documentation

The full documentation lives in [docs/en](docs/en/README.md). From a source
checkout, use the repo launcher to build or serve the book:

```bash
sipp docs build
sipp docs serve
```

`sipp docs` installs required mdBook tooling when missing and configures the
Mermaid assets used by the technical book. Without the launcher, use
`cargo xtask docs build` or `cargo xtask docs serve`.

---

## Technical Roadmap

Our core development trajectory is oriented around expanding the edge-cloud infrastructure for running hybrid systems, where local and cloud resources are orchestrated seamlessly.

For a detailed structural breakdown of milestones, memory architectures, and long-term research initiatives, see the full [Sipp Technical Roadmap](docs/en/roadmap.md).

---

## Maintainers & Contributors

To bootstrap the workspace, initialize a cross-platform profile, and inspect
the available test suites:

```bash
source ./setup.sh
sipp doctor
sipp test list
```

*(On Windows platforms, execute `.\setup.ps1` inside PowerShell or `setup.cmd` via classic CMD if not using Git Bash or WSL).*

The setup scripts install `sipp` as a repo-local alias for `cargo xtask`. If
the launcher is not active, replace `sipp` with `cargo xtask` in any command
below.

### Common Build and Run Tasks

```bash
sipp build wasm
sipp run examples serve browser

sipp build node --backend cpu
node examples/node/query.mjs <model.gguf> "Explain Sipp."

sipp build python --backend cpu
python examples/python/query.py <model.gguf> "Explain Sipp."

sipp build swift   # macOS with Xcode only

sipp run demos serve chat
```

For thorough verification steps, consult the
[Source Builds Documentation](docs/en/maintainers/source-builds.md) and the
[Testing Framework Suite](docs/en/testing.md).

---

## Repository Layout

* [crates](crates%2FREADME.md): The published core `sipp-rs` and low-level backend `sipp-sys` Rust crates.
* [lib](lib%2Fgateway%2FREADME.md): High-level language package surfaces and gateway proxy toolkit.
* [bindings](bindings%2FREADME.md): Native Node.js bindings, Python
  extensions, Swift UniFFI bindings, and browser-compiled WASM targets.
* [apps](apps%2FREADME.md): First-party user interfaces and monitoring implementations.
* [examples](examples%2FREADME.md): Small, runable framework integration blueprints.
* [demos](demos%2FREADME.md): Advanced browser sandboxes running on public package surfaces.
* [tools/playground](tools%2Fplayground%2FREADME.md): Live browser-runtime profiling and hardware execution diagnostics.
* `xtask/`: Internal cargo automation engine driving build, test, and package deployment pipelines.

## License

Sipp is licensed under the Apache-2.0 License. Vendored third-party
dependencies preserve their respective upstream open-source licensing
constraints and documentation requirements; see the
[third-party notices](THIRD_PARTY_NOTICES.md).
