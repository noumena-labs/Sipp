# Testing

CogentLM tests are cataloged by `cargo xtask test list`. Use that command first
when choosing a target or checking what CI runs.

## Commands

`cargo xtask test` has four top-level actions:

- `list`: list unit and smoke suites and optionally discover/search cheap cases.
- `unit`: run deterministic code-flow and API-layer tests.
- `smoke`: run holistic integration smoke tests by target.
- `verify`: analyze existing coverage artifacts and validate test structure.

## Common Commands

```bash
cargo xtask test list
cargo xtask test list --group unit --layer interface --cases --search router --format json
cargo xtask test unit
cargo xtask test unit whitebox
cargo xtask test unit interface
cargo xtask test unit xtask
cargo xtask test unit rust --package cogentlm-engine
cargo xtask test unit node --backend cpu
cargo xtask test unit python --backend cpu
cargo xtask test smoke node --backend cpu
cargo xtask test smoke provider-gateway
cargo xtask test smoke model --backend cpu
cargo xtask test smoke browser
cargo xtask test verify --target whitebox
cargo xtask test verify --changed
```

`test unit` owns deterministic tests. `whitebox` covers internal code-flow
suites, while `interface` covers deterministic public API and binding package
checks. Unit target names expose target-specific options, such as
`test unit rust --package <crate>` and `test unit node --backend cpu`.

`test smoke` owns holistic integration checks. Model-backed smoke targets
(`cli`, `rust`, `node`, `python`, and `model`) default to the setup sample model
cache under `.build/models` when `--model` is omitted. They accept `--backend`,
`--model`, `--offline`, `--prompt`, `--max-tokens`, and `--temperature`; Rust,
Node, and Python also accept repeated `--case query|chat`.
`provider-gateway` runs hermetic fake-provider gateway smoke tests without live
network calls or provider credentials.

`test unit` and `test smoke` write `.build/test/run-report.json` and
`.build/test/run-report.md`. Coverage-capable unit suites also write fresh
coverage artifacts under `.build/coverage/`.

`test verify` does not execute test suites. It validates test structure,
catalog ownership, test/runtime code separation, optional changed-file coverage,
and existing coverage artifacts.
