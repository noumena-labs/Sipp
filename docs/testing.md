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
cargo xtask test unit browser-package
cargo xtask test unit demos
cargo xtask test unit node --backend cpu
cargo xtask test unit python --backend cpu
cargo xtask test smoke suite example-node --backend cpu
cargo xtask test smoke suite benchmark-browser
cargo xtask test smoke group examples --backend cpu
cargo xtask test smoke group local-model --backend cpu
cargo xtask test smoke group full --backend cpu
cargo xtask test verify --target whitebox
cargo xtask test verify --changed
```

`test unit` owns deterministic tests. `whitebox` covers internal code-flow
suites, while `interface` covers deterministic public API and binding package
checks. Unit target names expose target-specific options, such as
`test unit rust --package <crate>` and `test unit node --backend cpu`.
Browser package tests live under `lib/web`; demo tests are
discovered under `demos`.

`test smoke` owns holistic integration checks. It is split into explicit
namespaces:

- `test smoke suite <name>` runs exactly one smoke suite.
- `test smoke group <name>` runs a named bundle of smoke suites.

Model-backed smoke suites default to the setup sample model cache under
`.build/models` when `--model` is omitted. Rust, Node, Python, and browser
example smoke accept repeated `--case query|chat`.

## Smoke Suites

| Command | What runs | Code location |
| --- | --- | --- |
| `cargo xtask test smoke suite cli` | Staged local CLI generation smoke | `apps/cli` |
| `cargo xtask test smoke suite example-rust` | Rust `query`/`chat` examples | `examples/rust` |
| `cargo xtask test smoke suite example-node` | Node `query.mjs`/`chat.mjs` examples | `examples/node` |
| `cargo xtask test smoke suite example-python` | Python `query.py`/`chat.py` examples | `examples/python` |
| `cargo xtask test smoke suite example-browser` | Browser `query.html`/`chat.html` examples through Playwright | `examples/web` |
| `cargo xtask test smoke suite benchmark-browser` | Browser runtime benchmark smoke through Playwright | `benchmarks/browser` |
| `cargo xtask test smoke suite provider-gateway` | Hermetic fake-provider gateway smoke | `crates/gateway`, `crates/gateway-providers` |
| `cargo xtask test smoke suite llama-backend-ops` | llama.cpp backend operation correctness smoke | `third_party/llama.cpp` |

## Smoke Groups

| Command | Suites |
| --- | --- |
| `cargo xtask test smoke group examples` | `example-rust`, `example-node`, `example-python`, and `example-browser` |
| `cargo xtask test smoke group local-model` | `cli`, `example-rust`, `example-node`, and `example-python` |
| `cargo xtask test smoke group full` | Every smoke suite, including benchmark, gateway, and llama checks |

Use `cargo xtask run examples serve browser` to manually serve browser examples,
and `cargo xtask run benchmarks serve browser` to manually serve the benchmark
app. Benchmark validation remains under `test smoke suite benchmark-browser`.

`test unit` and `test smoke` write `.build/test/run-report.json` and
`.build/test/run-report.md`. Coverage-capable unit suites also write fresh
coverage artifacts under `.build/coverage/`.

`test verify` does not execute test suites. It validates test structure,
catalog ownership, test/runtime code separation, optional changed-file coverage,
and existing coverage artifacts.

## Package Locations

- `lib/web` publishes `@noumena-labs/cogentlm` and public `cogentlm`.
- `lib/node` publishes `@noumena-labs/cogentlm-server` and public `cogentlm-server`.
- `lib/python` publishes Python `cogentlm`.
- `lib/rust` is the Rust facade crate used by Rust applications and examples.
