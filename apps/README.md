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

See [../docs/en/reference/cli.md](../docs/en/reference/cli.md),
[../docs/en/gateway/server.md](../docs/en/gateway/server.md), and
[../docs/en/gateway/docker.md](../docs/en/gateway/docker.md) for application
reference docs.
