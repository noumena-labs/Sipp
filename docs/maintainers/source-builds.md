# Source Builds

Use the source checkout when developing CogentLM itself, validating package
artifacts, running examples, or deploying the gateway server before a public
server artifact exists.

## Bootstrap

From the repository root:

```bash
source ./setup.sh
clm doctor
clm test list
```

On Windows, run `.\setup.ps1` from PowerShell or `setup.cmd` from CMD. After
setup, `clm` is a repo-local alias for `cargo xtask`; use `cargo xtask ...`
with the same arguments if the launcher is not active.

## Build Targets

Use the xtask orchestrator instead of direct build commands when compiling
CogentLM targets. It manages native dependencies, backend toolchains, and
package staging.

```bash
clm build core
clm build node --backend cpu
clm build python --backend cpu
clm build gateway-server --backend cpu
clm build wasm
clm build all
```

Use `--backend vulkan`, `--backend cuda`, `--backend metal`, or
`--backend all` where a native package target supports those backends.

## Examples And Demos

Run browser examples and demos through `clm`. These commands start Vite dev
servers and do not accept native backend flags:

```bash
clm run examples serve browser
clm run demos serve avatar
clm run demos serve simulation
```

## Gateway Hello World Examples

Gateway example workflows start a local gateway, run a client example, and stop
the gateway when the client exits. They start `examples/gateway` and then run a
client from `examples/rust`, `examples/node`, or `examples/python`.

Use `--case query|chat|embed` to choose the client case. Use
`--backend cpu|vulkan|cuda|metal` when the gateway process should use a
specific native backend.

```bash
clm run examples gateway rust --case query
clm run examples gateway node --case chat
clm run examples gateway python --case embed --backend vulkan
```

## Playground

The browser playground lives under `tools/playground`. Use it to inspect local
inference, vision model setup, GGUF loading, runtime observability, and
repeatable browser runtime smoke checks.

```bash
clm run tools serve playground
```

## Gateway Server

The release workflow does not yet publish a standalone gateway-server binary or
container image. Use `clm` for source checkout checks and raw Docker commands
for container deployment. The canonical source guide is
[Gateway Server](../gateway/server.md); Docker deployment is covered in
[Gateway Docker](../gateway/docker.md).

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
clm build gateway-server --backend cpu
cp apps/gateway-server/config/local.toml.example apps/gateway-server/config/local.toml
clm run gateway-server check --config apps/gateway-server/config/local.toml
clm run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

The copied local config expects a local GGUF model under `.build/models` and a
literal `admin_password` in the selected TOML file. Keep production TOML files
private because they contain the Admin Dashboard password.

## Validation

Use the narrowest relevant target from [Testing](../testing.md). Common
entry points are:

```bash
clm test list
clm test unit group full
clm test smoke group examples --backend cpu
clm test verify --target public-docs
```
