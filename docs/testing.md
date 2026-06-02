# Testing

CogentLM tests are cataloged by `cargo xtask test list`. Use that command first
when choosing a suite or checking what CI runs.

## Commands

`cargo xtask test` has four top-level actions:

- `list`: list suites and optionally discover/search individual cases.
- `run`: execute tests selected by suite or category and write run/coverage artifacts.
- `verify`: analyze existing coverage artifacts and validate test structure.
- `help`: show detailed CLI help through Clap's built-in help command.

## Common Commands

```bash
cargo xtask test list
cargo xtask test list --cases
cargo xtask test list --category whitebox --cases --search router --format json
cargo xtask test run
cargo xtask test run --category whitebox
cargo xtask test run --suite xtask
cargo xtask test run --suite rust-crates --package cogentlm-engine
cargo xtask test run --suite node-package --backend cpu
cargo xtask test verify --category whitebox
```

`--suite` can be repeated on `list`, `run`, and `verify`. For `test run`,
`--backend`, `--model`, and `--offline` only affect suites that build native
bindings or run model-backed checks. They are ignored by xtask and plain
Rust/package listing paths.

`test run` is the only test command that executes suites. It writes
`.build/test/run-report.json`, `.build/test/run-report.md`, and fresh coverage
artifacts under `.build/coverage/` for coverage-capable suites.

`test verify` does not execute test suites. It validates test structure,
catalog ownership, test/runtime code separation, optional changed-file coverage,
and existing coverage artifacts.
