# Source Builds

Use the source checkout when developing Sipp itself, validating package
artifacts, running examples, or deploying the gateway server before a public
server artifact exists.

## Bootstrap

From the repository root:

```bash
source ./setup.sh
sipp doctor
sipp test list
```

On Windows, run `.\setup.ps1` from PowerShell or `setup.cmd` from CMD. After
setup, `sipp` is a repo-local alias for `cargo xtask`; use `cargo xtask ...`
with the same arguments if the launcher is not active.

## Build Targets

Use the xtask orchestrator instead of direct build commands when compiling
Sipp targets. It manages native dependencies, backend toolchains, and
package staging.

```bash
sipp build core
sipp build node --backend cpu
sipp build python --backend cpu
sipp build swift
sipp build gateway-server --backend cpu
sipp build wasm
sipp build all
```

Use `--backend vulkan`, `--backend cuda`, `--backend metal`, or
`--backend all` where a native package target supports those backends.

The Swift target requires macOS and Xcode. It produces one XCFramework for
macOS and iOS. macOS arm64 and iOS device arm64 use Metal; macOS x86_64 and
iOS Simulator arm64/x86_64 use CPU. The backend is fixed per slice, with no
build selector or runtime fallback. The build also creates the macOS
command-line example and the ad-hoc-signed sandboxed SwiftUI app under
`.build/artifacts/swift`.

## Swift Distribution

[`noumena-labs/Sipp-Swift`](https://github.com/noumena-labs/Sipp-Swift) is the
CI-generated Swift Package Manager distribution for Sipp. It is a delivery
repository, not a second development repository, and its generated package
files must not be edited manually.

Sipp is the only source of truth. Swift sources and tests live under
`lib/swift`, Rust and UniFFI binding sources live under `bindings/swift`, and
`xtask` owns XCFramework assembly and package validation. The main CI workflow
tests the Swift package without publishing. Dev and stable release workflows
build and validate the XCFramework, stage the package, record the exact source
revision, and push the generated snapshot to `Sipp-Swift`.

Sipp tags are the sole version authority. A dev version such as `0.1.4-dev1`
produces the Swift tag `0.1.4-dev1`, while a stable version such as `0.1.4`
produces `0.1.4`. Existing Swift tags are checked only to make retries
idempotent and prevent replacement of an immutable release.

The XCFramework archive is stored as a GitHub release asset rather than
committed to Git. The generated `Package.swift` points its binary target to the
versioned archive and checksum, allowing consumers to use Sipp as a normal
SwiftPM dependency.

CUDA builds compile a portable cloud GPU architecture list by default. Set
`SIPP_CUDA_ARCHITECTURES` (semicolon-separated CMake entries, for example
`80` for A100 only) before building to narrow the list for faster local
builds. See [docs/gateway/docker.md](../gateway/docker.md) for the full list
and rationale.

## Examples And Demos

Run browser examples and demos through `sipp`. These commands start Vite dev
servers and do not accept native backend flags:

```bash
sipp run examples serve browser
sipp run demos serve avatar
sipp run demos serve simulation
```

## Gateway Hello World Examples

Gateway example workflows start a local gateway, run a client example, and stop
the gateway when the client exits. They start `examples/gateway` and then run a
client from `examples/rust`, `examples/node`, or `examples/python`.

Use `--case query|chat|embed` to choose the client case. Use
`--backend cpu|vulkan|cuda|metal` when the gateway process should use a
specific native backend.

```bash
sipp run examples gateway rust --case query
sipp run examples gateway node --case chat
sipp run examples gateway python --case embed --backend vulkan
```

## Playground

The browser playground lives under `tools/playground`. Use it to inspect local
inference, vision model setup, GGUF loading, runtime observability, and
repeatable browser runtime smoke checks.

```bash
sipp run tools serve playground
```

## Gateway Server

The release workflow does not yet publish a standalone gateway-server binary or
container image. Use `sipp` for source checkout checks and raw Docker commands
for container deployment. The canonical source guide is
[Gateway Server](../gateway/server.md); Docker deployment is covered in
[Gateway Docker](../gateway/docker.md).

```bash
cp apps/gateway-server/config/local.toml.example apps/gateway-server/config/local.toml
cp apps/gateway-server/.env.example apps/gateway-server/.env
set -a
. apps/gateway-server/.env
set +a
sipp run gateway-server check --config apps/gateway-server/config/local.toml --backend cpu
sipp run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

The copied local config expects a local GGUF model under `.build/models` and a
dashboard password env var named by the selected TOML file. Keep secrets env
files private because they contain the Admin Dashboard password and provider
credentials.

## Validation

Use the narrowest relevant target from [Testing](../testing.md). Common
entry points are:

```bash
sipp test list
sipp test unit group full
sipp test smoke group examples --backend cpu
sipp test verify --target public-docs
```
