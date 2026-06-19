
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
</div>



> [!WARNING]
> Sipp is under active development. Breaking changes are expected as we optimize the runtime layers. It might not be suitable for mission-critical production environments yet. If you find issues, bugs, or missing features, please open a GitHub issue.

### [Read the documentation →](docs/en/README.md)

## What is Sipp?

Sipp is an all-in-one, high-performance AI framework for building web, desktop, and edge applications. It ships as a cohesive SDK with a unified, symmetric API for local, provider, and cloud gateway inference.

At its core is **Sipp Engine**, a blazing-fast runtime built to run anywhere: in the browser, on the desktop, or on bare-metal cloud infrastructure. Written in Rust, C++, and `ggml`, it delivers low startup times and a minimal memory footprint.

```javascript
import { SippClient } from '@sipp/sipp';
const blender = new SippClient();

// 1. Initialize high-speed, local WebGPU or CUDA inference
const juice = await blender.add('edge', { kind: 'local', source: '/models/llama3.gguf' });

// 2. Or connect to a secure cloud proxy using the exact same interface
const ice = await blender.add('cloud', { kind: 'gateway', baseUrl: 'https://gateway.example.com/v1/' });

// Run inference on either endpoint seamlessly with a symmetric API
const [smoothie, snowcone] = await Promise.all([
  blender.chat([{ role: 'user', content: 'Explain Sipp.' }], { endpoint: juice }),
  blender.chat([{ role: 'user', content: 'Create a Sipp app.' }], { endpoint: ice })
]);
```

The unified SDK lets you dynamically partition and optimize complex application logic between local and cloud compute. Instead of wrestling with fragmented web runtimes, disconnected native wrappers for desktop, or custom middleware to protect API keys, you only need Sipp.

It packages a **high-performance WebGPU engine**, with a secure container gateway proxy into a single, neat toolkit. Future releases will focus on embedded vector memory, on-device PII masking, and automated smart routing. See [Roadmap](docs/en/roadmap.md).

```bash
sipp build wasm                # Compile high-performance WebGPU assets
sipp run demos serve chat      # Launch a local, hardware-accelerated test canvas

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
npm install @sipp/sipp

# For Node.js backend deployments (with native CUDA/Metal compilation)
npm install @sipp/sipp-server

# For native systems development and application embedding
cargo add sipp-rs

# For Python automation and data engineering pipelines
# (sippy wheels ship from GitHub Releases today; full PyPI build matrix in progress)
# pip install sipppy

# Deploy the secure cloud gateway server instance via Docker
# (cloud gateway will be available in the future, currently building from source)
# docker pull noumena/sipp-gateway

```

---

## Runtimes & Flavors

Most developers should start with our pre-built, published packages rather than compiling directly from the monorepo source.

| Surface | Module | Install | Docs |
| --- | --- | --- | --- |
| **Browser** | Sipp Edge | `npm install @sipp/sipp` | [Browser package](docs/en/packages/browser.md) |
| **Node.js** | Sipp Core | `npm install @sipp/sipp-server` | [Node.js package](docs/en/packages/node.md) |
| **Rust** | Sipp Core | `cargo add sipp-rs` | [Rust package](docs/en/packages/rust.md) |
| **Python** | Sipp Core | Wheels available on release page | [Python package](docs/en/packages/python.md) |
| **Gateway Server** | Sipp Cloud | Source-built | [Gateway Server](docs/en/gateway/server.md) |
| **Gateway Toolkit** | Sipp Cloud | Source-built | [Gateway toolkit](docs/en/gateway/toolkit.md) |

---

## Quick Starts

### 1. Edge Quick Start (Hardware-Accelerated Client Inference)

Initialize the local engine client to execute model weights directly on the client machine's shader cores using WebGPU.

```bash
npm install @sipp/sipp

```

```javascript
import { Client } from '@sipp/sipp';

const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain Sipp in one sentence.' },
];

const client = new Client();
const endpoint = await client.add('default', {
  kind: 'local',
  source: '/models/model.gguf',
});

const run = client.chat(messages, {
  endpoint,
  maxTokens: 64,
});

console.log((await run.response).text);
await client.close();

```

### 2. Cloud Gateway Quick Start (Preemptive Cloud Proxying)

Cloud gateway clients use the exact same `Client` API layout. The gateway owns model paths, provider credentials, access policies, and centralized metrics tracking; your client application code only needs the gateway routing target URL.

```javascript
import { Client } from '@sipp/sipp';

const client = new Client();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'upstream-cluster',
  baseUrl: 'https://gateway.example.com/v1/',
  authentication: { kind: 'bearer', value: await getGatewayToken() },
});

const run = client.query('Explain gateway inference.', {
  endpoint,
  maxTokens: 64,
});

console.log((await run.response).text);
await client.close();

```

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

The full documentation lives in [docs/en](docs/en/README.md). From a source checkout, use the `sipp docs` CLI tool utility to build or serve the book resource:

```bash
sipp docs build
sipp docs serve

```

`sipp docs` automatically evaluates and installs required mdBook tooling when missing and configures the Mermaid compilation assets used by the technical book layout.

---

## Technical Roadmap

Our core development trajectory is oriented around expanding the edge-cloud infrastructure for running hybrid systems, where local and cloud resources are orchestrated seamlessly.

For a detailed structural breakdown of milestones, memory architectures, and long-term research initiatives, see the full [Sipp Technical Roadmap](docs/en/roadmap.md).

---

## Maintainers & Contributors

To bootstrap the workspace workspace environment, initialize cross-platform profiles, and run structural unit assertions, utilize the integrated CLI environment scripts:

```bash
source ./setup.sh
sipp doctor
sipp test list

```

*(On Windows platforms, execute `.\setup.ps1` inside PowerShell or `setup.cmd` via classic CMD if not using Git Bash or WSL).*

### Common Architecture Compilation Tasks:

```bash
sipp build wasm && sipp run examples serve browser
sipp build node --backend cpu && node examples/node/query.mjs <model.gguf> "Explain Sipp."
sipp build python --backend cpu && python examples/python/query.py <model.gguf> "Explain Sipp."
sipp run demos serve chat

```

For thorough verification steps, consult the [Source Builds Documentation](docs%2Fmaintainers%2Fsource-builds.md) and our full [Testing Framework Suite](docs%2Ftesting.md).

---

## Repository Layout

* [crates](crates%2FREADME.md): The published core `sipp-rs` and low-level backend `sipp-sys` Rust crates.
* [lib](lib%2Fgateway%2FREADME.md): High-level language package surfaces and gateway proxy toolkit.
* [bindings](bindings%2FREADME.md): Native Node.js bindings, Python extensions, and browser-compiled WASM targets.
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
