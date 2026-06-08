# Simulation Demo

`demos/simulation` is a multi-agent browser simulation built with React,
three.js, the browser package, and the director helpers. The default Banana
Dash scenario runs multiple local model-backed agents plus a director that
narrates and adjudicates contested events.

## Run

```bash
cargo xtask run demos serve simulation
```

Open the printed local URL, load the default model, then press `Start`.

## What It Demonstrates

- A shared `CogentClient` driving multiple local model-backed brains.
- Director config loading from `public/directors/courtyard/director.json`.
- Runtime event tracing for agent and director decisions.
- A deterministic simulation loop combined with model-authored choices.
- Manual pause, step, reset, and scenario sizing controls.

Unit coverage for the simulation runtime is included in the demos test suite:

```bash
cargo xtask test unit suite demos
```

See [../../docs/examples-demos.md](../../docs/examples-demos.md) for the demo
index.
