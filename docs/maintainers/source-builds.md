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

## Examples
Browser examples and demos. Run examples through the `clm` commands. Use `--backend <backend>` to specify the backend to use. 

```bash
clm run examples serve browser --backend vulkan 
clm run demos serve avatar --backend cuda 
clm run demos serve simulation --backend metal 
clm run demos serve simulation
```

## Run Gateway Toy Examples

Gateway example workflows start a local gateway, run a client example, and stop
the gateway when the client exits. This will both start the `examples/gateway` and start a client `examples/rust`, `examples/node`, or `examples/python` to use the gateway. The `--case <case>` flag is used to specify the case to run. The available cases are `query`, `chat`, and `embed`. The `--backend <backend>` flag is used to specify the backend to use. 

```bash
clm run examples gateway rust --case query
clm run examples gateway node --case chat
clm run examples gateway python --case embed
```

## Run Playground 

The playground is a web application that allows you to interact with CogentLM. It is served locally from the `apps/playground` directory. It will show you how local inference works, how vision models work, and how to set up GGUF files. It intentially built for getting observability into how the library works.

```bash
clm run tools serve playground
```

# Gateway Server 

The release workflow does not yet publish a standalone gateway-server binary or
container image. Use `clm` for source checkout checks and raw Docker commands
for container deployment. The canonical source guide is
[Gateway Server](../gateway/server.md); Docker deployment is covered in
[Gateway Docker](../gateway/docker.md).

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
clm build gateway-server --backend cpu
clm run gateway-server check --config apps/gateway-server/config/development.toml
clm run gateway-server serve --config apps/gateway-server/config/development.toml --backend cpu
```

The source development config expects a local GGUF model under `.build/models`
and a literal `admin_password` in the selected TOML file. Keep production TOML
files private because they contain the Admin Dashboard password.

# Validation

Use the narrowest relevant target from [Testing](../testing.md). Common
entry points are:

```bash
clm test list
clm test unit group full
clm test smoke group examples --backend cpu
clm test verify --target public-docs
```
