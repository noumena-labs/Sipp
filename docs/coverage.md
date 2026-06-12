# Coverage

CogentLM coverage is driven through the same test catalog used by
`cargo xtask test list`. General test command guidance lives in
[testing.md](testing.md).

## Commands

```bash
cargo xtask test list
cargo xtask test list --group unit --layer whitebox --cases --format json
cargo xtask test unit group whitebox
cargo xtask test verify --target whitebox
cargo xtask test verify --target node
cargo xtask test verify --changed
```

`test unit` is the command that executes deterministic coverage-capable suites
and creates fresh coverage data. Rust writes coverage through `cargo-llvm-cov`,
Node writes coverage through `c8`, and Python writes coverage through
`pytest-cov`.

`test verify` defaults to all coverage-capable unit suites. It does not execute
test suites, build bindings, download models, or run smoke tests. Use
`--target` to narrow which existing coverage artifacts are analyzed. Explicitly
selecting a unit target that is not coverage-capable fails with a clear error.

`--changed` validates that changed first-party source files owned by the
selected unit suites have matching changed tests owned by the same catalog
suites. `test verify` also checks catalog ownership and test/runtime code
separation so tests do not live inside runtime source files.

`test list --format json` is the stable catalog surface used by CI and
contributors. Each suite entry includes `id`, `group`, `layer`, `description`,
`requirements`, `sourceRoots`, `backendPolicy`, `coverage`, and
`caseDiscovery`. Use `--cases` when a tool needs discoverable files and case
names that map to the suite runner.

## Tools

Coverage reporting uses the tools required by the selected report areas:

- `cargo-llvm-cov` for Rust/native execution and report rendering.
- `c8` for Node wrapper coverage during `test unit suite node-package`.
- `pytest-cov` for Python wrapper coverage during `test unit suite python-package`.

`test verify` only reads existing coverage artifacts and renders summaries from
them.

## Outputs

Reports are written under `.build/coverage/`:

- `rust/lcov.info` and `rust/html/`
- `node/lcov.info`
- `python/lcov.info`, `python/cobertura.xml`, and `python/html/`
- `baseline.json`
- `coverage-summary.md`

Test command reports are written under `.build/test/`:

- `run-report.json` and `run-report.md`
- `verify-report.json` and `verify-report.md`

The baseline includes first-party `crates/` and `bindings/` code. It
intentionally excludes generated outputs, caches, tests, examples,
`third_party/`, and the vendored `crates/sys/llama.cpp/` tree.

## Policy

The current implementation records the baseline and does not fail on percentage
thresholds. It does fail when an enabled coverage area produces an empty
first-party report. Thresholds should be added after the baseline is stable and
the largest uncovered first-party areas are addressed.
