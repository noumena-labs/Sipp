# Applications

`apps/` contains first-party applications built from the public Sipp crates
and packages. Applications own command-line behavior, configuration files,
HTTP routes, listeners, deployment policy, and user-facing defaults.

## Applications

- [`cli`](cli/README.md): command-line local GGUF text generation.
- [`gateway-server`](gateway-server/README.md): first-party HTTP gateway server
  with TOML, bearer tokens, target policy, probes, metrics, and deployment
  behavior.

## Build

Use xtask from the repository root:

```bash
cargo xtask build cli --backend cpu
cargo xtask build core
```

See [../docs/reference/cli.md](../docs/reference/cli.md),
[../docs/gateway/server.md](../docs/gateway/server.md), and
[../docs/gateway/docker.md](../docs/gateway/docker.md) for application
reference docs.
