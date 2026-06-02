# Coverage

CogentLM coverage is driven through the same test catalog used by `cargo xtask test list`.
General test profile guidance lives in [testing.md](testing.md).

## Commands

```bash
cargo xtask test list
cargo xtask test list --category whitebox --cases --format json
cargo xtask test all --profile contributor
cargo xtask test coverage --scope whitebox --backend cpu
cargo xtask test coverage --scope all --backend cpu
```

`--scope whitebox` collects Rust/native coverage for first-party crates and Rust binding crates. `--scope all` also runs interface-oriented Node and Python wrapper coverage plus browser/model interface smokes.

`test list --format json` is the stable catalog surface used by CI and contributors. Each suite entry includes `id`, `category`, `description`, `profiles`, `requirements`, `backendPolicy`, `coverage`, and `caseDiscovery`. Use `--cases` when a tool needs the discoverable files and case names that map to the suite runner.

## Tools

Coverage requires:

- `cargo-llvm-cov` for Rust/native reports.
- `c8` for Node wrapper reports.
- `pytest-cov` for Python wrapper reports.

The CI coverage workflow installs `cargo-llvm-cov`; Node and Python coverage tools are declared in the binding package dev dependencies.

## Outputs

Reports are written under `.build/coverage/`:

- `rust/lcov.info` and `rust/html/`
- `node/lcov.info`
- `python/lcov.info`, `python/cobertura.xml`, and `python/html/`
- `baseline.json`
- `coverage-summary.md`

The baseline includes first-party `crates/` and `bindings/` code. It intentionally excludes `packages/`, `apps/`, generated outputs, caches, tests, examples, and `third_party/`.

## Policy

The first implementation records the baseline and does not fail on percentage thresholds. It does fail when an enabled coverage area produces an empty first-party report: Rust/native for `--scope whitebox`, and Rust/native plus Node/Python wrappers for `--scope all`. Thresholds should be added after the baseline is stable and the largest uncovered first-party areas are addressed.

Public contributor CI uses `cargo xtask test all --profile contributor`, which avoids private submodules, sample model downloads, browser toolchains, and GPU/backend requirements. Internal CI and scheduled workflows keep the broader native, WASM, interface, model, and coverage gates.
