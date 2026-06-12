# Gateway Server

The Sipp Gateway Server is the first-party HTTP application for teams that
want one inference boundary for local GGUF targets and provider-backed targets.
It lives in `apps/gateway-server`.

This page covers source checkout and generated executable operation. Use
[Docker](docker.md) for container workflows and [Configuration](configuration.md)
for the TOML schema.

The current release workflow does not publish a standalone binary, public
container image, or `cargo install` target. Build it from the source checkout.

## Source Workflow

Use `sipp` for source checkout workflows. `sipp` is the setup-installed launcher
for `cargo xtask`; when the launcher is unavailable, use `cargo xtask` with
the same arguments.

```bash
cp apps/gateway-server/config/local.toml.example apps/gateway-server/config/local.toml
cp apps/gateway-server/.env.example apps/gateway-server/.env
set -a
. apps/gateway-server/.env
set +a
sipp run gateway-server check --config apps/gateway-server/config/local.toml --backend vulkan
sipp run gateway-server serve --config apps/gateway-server/config/local.toml --backend vulkan
```

Before running real on-board inference tests, update the ignored local TOML
with the token env names, admin password env name, and model path. Update only
secret values in the secrets env file.

`sipp run gateway-server check` builds the staged gateway distribution for the
selected backend, then runs `sipp-gateway check`. The binary `check`
command parses and validates TOML only. It does not read bearer-token
environment variables, load model files, contact providers, or bind ports.

`sipp run gateway-server serve` builds the staged gateway distribution, then
runs the generated `sipp-gateway` executable from the workspace root. It
reads secret environment variables named by TOML, loads targets, binds both
listeners, and exits cleanly on Ctrl-C.

Use `--backend cpu|vulkan|cuda|metal|all` to select the backend compiled into
the staged gateway distribution.

## Provider-Only Source Workflow

Provider-only gateways route to upstream APIs and do not load a local GGUF
model. Use a CPU gateway build because inference happens at the provider:

```bash
cp apps/gateway-server/config/provider-only.toml.example apps/gateway-server/config/provider-only.toml
cp apps/gateway-server/.env.example apps/gateway-server/.env
set -a
. apps/gateway-server/.env
set +a
sipp run gateway-server check --config apps/gateway-server/config/provider-only.toml --backend cpu
sipp run gateway-server serve --config apps/gateway-server/config/provider-only.toml --backend cpu
```

Use [Configuration](configuration.md) for Anthropic and OpenAI-compatible
target snippets.

## Generated Executable

`sipp build gateway-server --backend <backend>` stages a runnable distribution
in `.build/artifacts/gateway-server`. The directory contains the
`sipp-gateway` executable, base runtime libraries, and selected GGML
backend plugins. The build also compiles the React Admin Dashboard from
`apps/gateway-server/admin-ui` and copies its Vite output to
`.build/artifacts/gateway-server/admin-ui`. Keep the executable, dashboard
asset directory, and runtime libraries together.

Direct execution must put the artifact directory on the dynamic loader path.
The executable reads dashboard assets from `admin-ui` beside the binary unless
`SIPP_GATEWAY_ADMIN_ASSETS_DIR` points at another Vite `dist` directory.

Linux:

```bash
set -a
. apps/gateway-server/.env
set +a
export LD_LIBRARY_PATH="$(pwd)/.build/artifacts/gateway-server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.build/artifacts/gateway-server/sipp-gateway check --config apps/gateway-server/config/local.toml
.build/artifacts/gateway-server/sipp-gateway serve --config apps/gateway-server/config/local.toml
```

macOS:

```bash
set -a
. apps/gateway-server/.env
set +a
export DYLD_LIBRARY_PATH="$(pwd)/.build/artifacts/gateway-server${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
.build/artifacts/gateway-server/sipp-gateway check --config apps/gateway-server/config/local.toml
.build/artifacts/gateway-server/sipp-gateway serve --config apps/gateway-server/config/local.toml
```

Windows PowerShell:

```powershell
Get-Content apps\gateway-server\.env | ForEach-Object {
    if ($_ -and -not $_.StartsWith("#")) {
        $name, $value = $_.Split("=", 2)
        Set-Item -Path "Env:$name" -Value $value
    }
}
$dist = Join-Path (Get-Location) ".build\artifacts\gateway-server"
$env:PATH = "$dist;$env:PATH"
.\.build\artifacts\gateway-server\sipp-gateway.exe check --config apps\gateway-server\config\local.toml
.\.build\artifacts\gateway-server\sipp-gateway.exe serve --config apps\gateway-server\config\local.toml
```

Relative `model` paths in TOML are resolved from the process working
directory. The `sipp run gateway-server ...` workflow runs from the workspace
root. When running the executable from another directory, use absolute model
paths or start the process from the workspace root.

## Backends

The gateway server supports the same native backend names as other native
targets:

- `cpu`: provider-only router build or local-inference diagnostic backend.
- `cuda`: NVIDIA CUDA backend.
- `metal`: Apple Metal backend on macOS.
- `vulkan`: Vulkan backend.
- `all`: host-supported backend set for build commands.

For on-board local target TOML, `backend = "auto"` selects the best compiled
and available backend in this order: CUDA, Metal, Vulkan, then CPU. Production
model-serving configs should use `auto` or an explicit GPU backend. Explicit
`cpu` disables GPU offload and is intended only for diagnostics. Explicit GPU
backends fail if that backend was not compiled or is unavailable.

## Admin Dashboard

The Admin Dashboard password is read from the env var named by TOML:

```toml
admin_password_env = "SIPP_GATEWAY_ADMIN_PASSWORD"
```

Keep the real value in a secrets env file or production secret manager.

## Related Docs

- [Docker](docker.md)
- [Configuration](configuration.md)
- [Testing](testing.md)
- [Operations](operations.md)
