# Gateway Server

The CogentLM Gateway Server is the first-party HTTP application for teams that
want one inference boundary for local GGUF targets and provider-backed targets.
It lives in `apps/gateway-server`.

This page covers source checkout and generated executable operation. Use
[Docker](docker.md) for container workflows and
[Configuration](configuration.md) for the TOML schema.

The current release workflow does not publish a standalone binary, public
container image, or `cargo install` target. Build it from the source checkout.

## Source Workflow

Use `clm` for source checkout workflows. `clm` is the setup-installed launcher
for `cargo xtask`; when the launcher is unavailable, use `cargo xtask` with
the same arguments.

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
clm build gateway-server --backend cpu
clm run gateway-server check --config apps/gateway-server/config/development.toml
clm run gateway-server serve --config apps/gateway-server/config/development.toml --backend cpu
```

Before running real local tests, copy the development TOML to an ignored local
file and set the literal `admin_password`, token env names, and model path:

```bash
cp apps/gateway-server/config/development.toml apps/gateway-server/config/local.toml
clm run gateway-server check --config apps/gateway-server/config/local.toml
clm run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

`clm run gateway-server check` builds the staged gateway distribution for the
selected backend, then runs `cogentlm-gateway check`. The binary `check`
command parses and validates TOML only. It does not read bearer-token
environment variables, load model files, contact providers, or bind ports.

`clm run gateway-server serve` builds the staged gateway distribution, then
runs the generated `cogentlm-gateway` executable from the workspace root. It
reads token environment variables, loads targets, uses `admin_password` from
TOML, binds both listeners, and exits cleanly on Ctrl-C.

## Generated Executable

`clm build gateway-server --backend <backend>` stages a runnable distribution
in `.build/artifacts/gateway-server`. The directory contains the
`cogentlm-gateway` executable, base runtime libraries, and selected GGML
backend plugins. Keep the executable and its companion files together.

Direct execution must put the artifact directory on the dynamic loader path.

Linux:

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
export LD_LIBRARY_PATH="$(pwd)/.build/artifacts/gateway-server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.build/artifacts/gateway-server/cogentlm-gateway check --config apps/gateway-server/config/local.toml
.build/artifacts/gateway-server/cogentlm-gateway serve --config apps/gateway-server/config/local.toml
```

macOS:

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
export DYLD_LIBRARY_PATH="$(pwd)/.build/artifacts/gateway-server${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
.build/artifacts/gateway-server/cogentlm-gateway check --config apps/gateway-server/config/local.toml
.build/artifacts/gateway-server/cogentlm-gateway serve --config apps/gateway-server/config/local.toml
```

Windows PowerShell:

```powershell
$env:COGENTLM_GATEWAY_TOKEN = "replace-me"
$dist = Join-Path (Get-Location) ".build\artifacts\gateway-server"
$env:PATH = "$dist;$env:PATH"
.\.build\artifacts\gateway-server\cogentlm-gateway.exe check --config apps\gateway-server\config\local.toml
.\.build\artifacts\gateway-server\cogentlm-gateway.exe serve --config apps\gateway-server\config\local.toml
```

Relative `model` paths in TOML are resolved from the process working
directory. The `clm run gateway-server ...` workflow runs from the workspace
root. When running the executable from another directory, use absolute model
paths or start the process from the workspace root.

## Backends

The gateway server supports the same native backend names as other native
targets:

- `cpu`: portable default backend.
- `cuda`: NVIDIA CUDA backend.
- `metal`: Apple Metal backend on macOS.
- `vulkan`: Vulkan backend.
- `all`: host-supported backend set for build commands.

For local target TOML, `backend = "auto"` selects the best compiled and
available backend in this order: CUDA, Metal, Vulkan, then CPU. Explicit
`cpu` disables GPU offload. Explicit GPU backends fail if that backend was not
compiled or is unavailable.

## Binds And Routes

In source and direct executable mode, `public_bind` and `management_bind` bind
directly on the host machine:

- The public listener serves `query`, `chat`, and `embed`.
- The management listener serves optional `index`, `health`, `readiness`,
  `metrics`, and password-protected `admin` routes.

For local development, bind both listeners to `127.0.0.1`. In production, keep
the management listener private or behind trusted access control.

## Admin Password

The Admin Dashboard password is configured directly in the TOML file:

```toml
admin_password = "replace-me"
```

`check` fails when the field is missing or blank. The dashboard uses the value
for login but never renders it. Because production TOML contains a secret,
keep real production config files private and out of source control.

## Related Docs

- [Docker](docker.md)
- [Configuration](configuration.md)
- [Testing](testing.md)
- [Operations](operations.md)

