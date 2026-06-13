
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
  <a href="docs/getting-started/quickstarts.md">Documentation</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://discord.gg/abzgfghhrq">Discord</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://github.com/noumena-labs/Sipp/issues">Issues</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="docs/roadmap.md">Roadmap</a>
  <br />
</div>



> [!WARNING]
> Sipp is under active development. Breaking changes are expected as we optimize the runtime layers. It might not be suitable for mission-critical production environments yet. If you find issues, bugs, or missing features, please open a GitHub issue.

### [Read the documentation →](docs/README.md)

## What is Sipp?

Sipp is an all-in-one, high-performance AI framework for building web, desktop, and edge applications. It ships as a single, cohesive library called `sipp`, providing a unified, symmetric API for local, provider, and cloud gateway inference.

At its core is **Sipp Engine**, a blazing-fast runtime built to run anywhere: in the browser, on the desktop, or on bare-metal cloud infrastructure. Written in Rust, C++, and `llama.cpp`, it delivers low startup times and a minimal memory footprint.

```javascript
import { SippClient } from 'sipp';
const blender = new SippClient();

// 1. Initialize high-speed, local WebGPU or CUDA inference
const juice = await blender.add('edge', { kind: 'local', source: '/models/llama3.gguf' });

// 2. Or connect to a secure cloud proxy using the exact same interface
const ice = await blender.add('cloud', { kind: 'gateway', baseUrl: 'https://gateway.example.com/v1/' });

// Run inference on either endpoint seamlessly with a symmetric API
const stream = await blender.chat([{ role: 'user', content: 'Explain Sipp.' }], { endpoint: juice });

```

The unified SDK lets you dynamically partition and optimize complex application logic between local and cloud compute. Instead of wrestling with fragmented web runtimes, disconnected native wrappers for desktop, or custom middleware to protect API keys, you only need `sipp`.

It packages a **high-performance WebGPU engine**, with a secure container gateway proxy into a single, neat toolkit. Future releases will focus on embedded vector memory, on-device PII masking, and automated smart routing. See [Roadmap](docs/roadmap.md).

```bash
sipp build wasm                # Compile high-performance WebGPU assets
sipp run demos serve chat      # Launch a local, hardware-accelerated test canvas

```

---

## Install

Sipp supports web browsers, desktop application wrappers, server environments, and native runtimes. Install the specific implementation layer for your surface environment:

```sh
# For Web Browsers, Next.js, and TanStack applications
npm install sipp

# For Node.js backend deployments (with native CUDA/Metal compilation)
npm install sipp-server

# For Python automation and data engineering pipelines
pip install sipp

# For native systems development and application embedding
cargo add sipp

# Deploy the secure cloud gateway server instance via Docker
docker pull noumena/sipp-gateway

```

---

## Runtimes & Flavors

Most developers should start with our pre-built, published packages rather than compiling directly from the monorepo source.

| Surface | Module | Install | Docs |
| --- | --- | --- | --- |
| **Browser** | Sipp Edge | `npm install sipp` | [Browser package](docs/packages/browser.md) |
| **Node.js** | Sipp Core | `npm install sipp-server` | [Node.js package](docs/packages/node.md) |
| **Python** | Sipp Core | `pip install sipp` | [Python package](docs/packages/python.md) |
| **Rust** | Sipp Core | `cargo add sipp` | [Rust package](docs/packages/rust.md) |
| **Gateway Server** | Sipp Cloud | Source-built | [Gateway Server](docs/gateway/server.md) |
| **Gateway Toolkit** | Sipp Cloud | Source-built | [Gateway toolkit](docs/gateway/toolkit.md) |

---

## Quick Starts

### 1. Edge Quick Start (Hardware-Accelerated Client Inference)

Initialize the local engine client to execute model weights directly on the client machine's shader cores using WebGPU.

```bash
npm install sipp

```

```javascript
import { Client } from 'sipp';

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
import { Client } from 'sipp';

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

The full documentation lives in [docs](docs/README.md). From a source checkout, use the `sipp docs` CLI tool utility to build or serve the book resource:

```bash
sipp docs build
sipp docs serve

```

`sipp docs` automatically evaluates and installs required mdBook tooling when missing and configures the Mermaid compilation assets used by the technical book layout.

---

## Technical Roadmap

Our core development trajectory is oriented around expanding the edge-cloud infrastructure for running hybrid systems, where local and cloud resources are orchestrated seamlessly.

For a detailed structural breakdown of milestones, memory architectures, and long-term research initiatives, see the full [Sipp Technical Roadmap](docs/roadmap.md).

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

* [crates](crates%2FREADME.md): The published core `sipp` and low-level backend `sipp-sys` Rust crates.
* [lib](lib%2Fgateway%2FREADME.md): High-level language package surfaces and gateway proxy toolkit.
* [bindings](bindings%2FREADME.md): Native Node.js bindings, Python extensions, and browser-compiled WASM targets.
* [apps](apps%2FREADME.md): First-party user interfaces and monitoring implementations.
* [examples](examples%2FREADME.md): Small, runable framework integration blueprints.
* [demos](demos%2FREADME.md): Advanced browser sandboxes running on public package surfaces.
* [tools/playground](tools%2Fplayground%2FREADME.md): Live browser-runtime profiling and hardware execution diagnostics.
* `xtask/`: Internal cargo automation engine driving build, test, and package deployment pipelines.

## License

Sipp is licensed under the Apache-2.0 License. Vendored third-party dependencies preserve their respective upstream open-source licensing constraints and documentation requirements.

