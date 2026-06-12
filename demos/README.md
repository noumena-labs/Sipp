# Demos

`demos/` contains browser experiences built on Sipp package surfaces. Use
them to inspect richer workflows, runtime behavior, and user-facing patterns.
Use `examples/` for smaller copyable integrations.

## Demos

- [`chat`](chat/README.md): browser chat interface for local GGUF models.
- [`avatar`](avatar/README.md): React and three.js character demo.
- [`proactive-ui`](proactive-ui/README.md): drawing-to-vision demo with
  runtime tracing.
- [`simulation`](simulation/README.md): multi-agent director simulation.

## Run

```bash
cargo xtask run demos serve chat
cargo xtask run demos serve avatar
cargo xtask run demos serve proactive-ui
cargo xtask run demos serve simulation
```

Demo tests live under:

```bash
cargo xtask test unit suite demos
```

See [../docs/examples-demos.md](../docs/examples-demos.md) for the
documentation index.
