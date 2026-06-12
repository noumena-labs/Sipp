<p align="center">
  <img src="docs/assets/sipp-logo-placeholder.svg" alt="Sipp logo placeholder" width="160">
</p>

<h1 align="center">Sipp</h1>

<p align="center">
  Local and gateway-backed inference runtimes for browser, Node.js, Python,
  and Rust applications.
</p>

<p align="center">
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/noumena-labs/Sipp/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/noumena-labs/Sipp/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/coverage.yml"><img alt="Coverage" src="https://github.com/noumena-labs/Sipp/actions/workflows/coverage.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/release.yml"><img alt="Release" src="https://github.com/noumena-labs/Sipp/actions/workflows/release.yml/badge.svg"></a>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue">
</p>

> [!WARNING]
> Sipp is under active development. Breaking changes are expected, so it
> might not be suitable for production environments yet.
> If you find issues, bugs, or missing features, please open a GitHub issue.

## Use Published Packages

Most developers should start with the published packages rather than building
from this repository.

| Surface | Install | Docs |
| --- | --- | --- |
| Browser | `npm install sipp` | [Browser package](docs/packages/browser.md) |
| Node.js | `npm install sipp-server` | [Node.js package](docs/packages/node.md) |
| Python | `pip install sipp` | [Python package](docs/packages/python.md) |
| Rust | `cargo add sipp` | [Rust package](docs/packages/rust.md) |
| Gateway Server | Source-built today | [Gateway Server](docs/gateway/server.md) |
| Gateway toolkit | Rust source artifact today | [Gateway toolkit](docs/gateway/toolkit.md) |

The current release workflow publishes browser npm, Node npm, Python wheel,
and Rust source artifacts. The gateway server is a user-facing deployment
surface, but it does not yet have a published binary, public container image,
or `cargo install` target.

## Browser Quick Start

```bash
npm install sipp
```

```js
import { SippClient } from 'sipp';

const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain Sipp in one sentence.' },
];

const client = new SippClient();
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

## Gateway Quick Start

Gateway clients use the same `SippClient` API. The gateway owns model paths,
provider credentials, access policy, and metrics; clients only need the gateway
URL, public target, and application-issued auth value.

```js
import { SippClient } from 'sipp';

const client = new SippClient();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: { kind: 'bearer', value: await getGatewayToken() },
});
const run = client.query('Explain gateway inference.', {
  endpoint,
  maxTokens: 64,
});
console.log((await run.response).text);
await client.close();
```

## Frameworks

- [Next.js](docs/packages/frameworks/nextjs.md): App Router route handlers,
  Client Components, gateway proxies, and streaming.
- [TanStack](docs/packages/frameworks/tanstack.md): TanStack Start server
  functions and TanStack Query patterns.
- [React And Vite](docs/packages/frameworks/vite-react.md): Browser package
  setup, WASM assets, OPFS model loading, and gateway examples.

## Documentation

The full documentation lives in [docs](docs/README.md). From a source checkout,
use `sipp docs` to build or serve the book:

```bash
sipp docs build
sipp docs serve
```

`sipp docs` installs the required mdBook tooling when missing and prepares the Mermaid assets used by the book.

Start with:

- [Installation](docs/getting-started/installation.md)
- [Quickstarts](docs/getting-started/quickstarts.md)
- [Using the Core Library](docs/packages/README.md)
- [Gateway Server](docs/gateway/server.md)
- [Frameworks](docs/packages/frameworks/README.md)
- [Known Issues](docs/known-issues.md)

## Maintainers

Use this source checkout for builds, examples, demos, package staging, and
tests. Bootstrap the repository from the workspace root:

```bash
source ./setup.sh
sipp doctor
sipp test list
```

On Windows, use `.\setup.ps1` in PowerShell or `setup.cmd` in CMD. The `sipp`
launcher is installed under `.build/bin` and forwards to `cargo xtask`; use
`cargo xtask ...` with the same arguments if the launcher is not active.

Common source workflows:

```bash
sipp build wasm && sipp run examples serve browser
sipp build node --backend cpu && node examples/node/query.mjs <model.gguf> "Explain Sipp."
sipp build python --backend cpu && python examples/python/query.py <model.gguf> "Explain Sipp."
sipp run demos serve chat
```

See [Source Builds](docs/maintainers/source-builds.md),
[Testing](docs/testing.md), and [Coverage](docs/coverage.md).

## Repository Layout

- [crates](crates/README.md): the published `sipp` and `sipp-sys` crates.
- [lib](lib/gateway/README.md): language package surfaces and gateway toolkit.
- [bindings](bindings/README.md): Node, Python, and browser WASM bindings.
- [apps](apps/README.md): first-party applications.
- [examples](examples/README.md): small, runnable integrations.
- [demos](demos/README.md): browser demos built on public package surfaces.
- [tools/playground](tools/playground/README.md): browser runtime diagnostics.
- `xtask/`: build, test, run, and packaging automation.

## License

Sipp is licensed under Apache-2.0. Vendored third-party components keep
their upstream licenses and documentation.
