# Gateway Server

The CogentLM Gateway Server is the first-party HTTP application for teams that
want one inference boundary for local GGUF targets and provider-backed targets.
This page covers source and generated-exe operation from a checkout. Use
[Gateway Server Docker](gateway-server-docker.md) for container workflows and
[Gateway Server Reference](../reference/gateway-server.md) for the TOML schema.

The current release workflow does not publish a standalone binary, public
container image, or `cargo install` target. Build it from the source checkout.

## Source Workflow

Use `clm` for source checkout workflows. `clm` is the setup-installed launcher
for `cargo xtask`; when the launcher is unavailable, use `cargo xtask` with the
same arguments.

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

`check` parses and validates TOML only. It does not read bearer-token
environment variables, load model files, contact providers, or bind ports.

`serve` first builds the staged gateway distribution for the requested backend,
then runs the generated `cogentlm-gateway` executable from the workspace root.
It reads token environment variables, loads targets, uses `admin_password` from
TOML, binds both listeners, and exits cleanly on Ctrl-C.

## Generated Executable

`clm build gateway-server --backend <backend>` stages the runnable distribution
in `.build/artifacts/gateway-server`. The executable depends on the runtime
libraries and backend plugins in that same directory, so direct execution must
put that directory on the dynamic loader path.

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
export LD_LIBRARY_PATH="$(pwd)/.build/artifacts/gateway-server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.build/artifacts/gateway-server/cogentlm-gateway check --config apps/gateway-server/config/local.toml
.build/artifacts/gateway-server/cogentlm-gateway serve --config apps/gateway-server/config/local.toml
```

Bash on macOS uses `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`.

Relative `model` paths in TOML are resolved from the process working directory.
The `clm run gateway-server ...` workflow runs from the workspace root. When
running the executable from another directory, use absolute model paths.

## Binds And Routes

In source/exe mode, `public_bind` and `management_bind` bind directly on the
host machine:

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
for login but never renders it. Because production TOML contains a secret, keep
real production config files private and out of source control.

## Related Docs

- [Gateway Server Docker](gateway-server-docker.md)
- [Gateway Server Reference](../reference/gateway-server.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Providers](../guides/providers.md)
