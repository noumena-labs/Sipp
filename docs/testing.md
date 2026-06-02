# Testing

CogentLM tests are cataloged by `cargo xtask test list`. Use that command first
when choosing a suite or checking what CI runs.

## Profiles

`cargo xtask test all --profile <profile>` runs a cumulative profile:

- `contributor`: `layout`, `xtask`
  Public-safe check for fork PRs. No private submodules, browser toolchains,
  sample model downloads, or GPU/backend requirements.
- `quick`: `contributor` + `rust-crates`
  Fast local Rust/core confidence check.
- `ci`: `quick` + `package-ts`, `rust-public-api`
  Internal pull-request and master gate.
- `full`: every cataloged suite
  Nightly/release validation, including bindings, app TypeScript, CLI, Node,
  Python, browser runtime, model smoke, and llama.cpp backend operation checks.

## Common Commands

```bash
cargo xtask test list
cargo xtask test list --cases
cargo xtask test all --profile contributor
cargo xtask test whitebox --suite rust-crates --package cogentlm-engine
cargo xtask test interface --suite node-package --backend cpu
cargo xtask test coverage --scope whitebox --backend cpu
```

`--backend`, `--model`, and `--offline` only affect suites that build native
bindings or run model-backed checks. They are ignored by layout, xtask, and
plain Rust/package listing paths.
