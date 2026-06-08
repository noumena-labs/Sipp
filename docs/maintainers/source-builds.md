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
clm build wasm
clm build all
```

Use `--backend vulkan`, `--backend cuda`, `--backend metal`, or
`--backend all` where a native package target supports those backends.

## Source Examples

```bash
cargo run -p cogentlm-rust-examples --bin query -- <model.gguf> "Explain local inference."
node examples/node/query.mjs <model.gguf> "Explain local inference."
python examples/python/query.py <model.gguf> "Explain local inference."
```

Gateway example workflows start a local gateway, run a client example, and stop
the gateway when the client exits:

```bash
clm run examples gateway rust --case query
clm run examples gateway node --case chat
clm run examples gateway python --case embed
```

Browser examples and demos:

```bash
clm run examples serve browser
clm run demos serve chat
clm run demos serve avatar
clm run demos serve simulation
clm run tools serve playground
```

## Gateway Server From Source

The release workflow does not yet publish a standalone gateway-server binary or
container image. Build or run it from the source checkout when deploying the
first-party gateway server:

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
cargo run -p cogentlm-gateway-server -- \
  check --config apps/gateway-server/config/production.toml
cargo run -p cogentlm-gateway-server -- \
  serve --config apps/gateway-server/config/production.toml
```

The checked-in Dockerfile and compose file are also source artifacts:

```bash
docker build -f apps/gateway-server/Dockerfile -t cogentlm-gateway:cpu .
docker compose -f apps/gateway-server/compose.yaml up
```

Keep the management listener private when deploying the server.

## Validation

Use the narrowest relevant target from [Testing](../testing.md). Common
entry points are:

```bash
clm test list
clm test unit group full
clm test smoke group examples --backend cpu
clm test verify --target public-docs
```
