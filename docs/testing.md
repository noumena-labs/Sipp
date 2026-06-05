# Testing

CogentLM tests are cataloged by `cargo xtask test list`. Use that command first
when choosing a target or checking what CI runs.

## Commands

`cargo xtask test` has four top-level actions:

- `list`: list unit and smoke suites and optionally discover/search cheap cases.
- `unit`: run deterministic code-flow and API-layer tests by suite or group.
- `smoke`: run holistic integration smoke tests by suite or group.
- `verify`: analyze existing coverage artifacts and validate test structure.

## Common Commands

```bash
cargo xtask test list
cargo xtask test list --group unit --layer interface --cases --search router --format json
cargo xtask test unit group full
cargo xtask test unit group whitebox
cargo xtask test unit group interface
cargo xtask test unit suite xtask
cargo xtask test unit suite rust-crates --package cogentlm-engine
cargo xtask test unit suite browser-package
cargo xtask test unit suite demos
cargo xtask test unit suite node-package --backend cpu
cargo xtask test unit suite python-package --backend cpu
cargo xtask test smoke suite example-node --backend cpu
cargo xtask test smoke suite benchmark-browser
cargo xtask test smoke group examples --backend cpu
cargo xtask test smoke group local-model --backend cpu
cargo xtask test smoke group full --backend cpu
cargo xtask test verify --target whitebox
cargo xtask test verify --changed
```

`test unit` owns deterministic tests. It is split into explicit namespaces:

- `test unit suite <name>` runs exactly one deterministic unit suite.
- `test unit group <name>` runs a named bundle of deterministic unit suites.

Unit suite names expose suite-specific options, such as
`test unit suite rust-crates --package <crate>` and
`test unit suite node-package --backend cpu`.

## Unit Suites

| Command | What runs | Code location |
| --- | --- | --- |
| `cargo xtask test unit suite xtask` | xtask CLI and orchestration tests | `xtask/src/tests` |
| `cargo xtask test unit suite rust-crates` | Core workspace crate unit tests | `crates`, `lib/rust` |
| `cargo xtask test unit suite rust-bindings` | Rust binding crate unit tests | `bindings/node`, `bindings/python`, `bindings/wasm` |
| `cargo xtask test unit suite browser-package` | Browser package TypeScript tests | `lib/web/tests` |
| `cargo xtask test unit suite demos` | Browser demo TypeScript tests | `demos` |
| `cargo xtask test unit suite api` | Crate-level public API integration tests | `lib/rust/tests`, `crates/*/tests` |
| `cargo xtask test unit suite cli` | CLI black-box integration tests | `apps/cli/tests` |
| `cargo xtask test unit suite node-package` | Deterministic Node package API tests | `lib/node`, `bindings/node` |
| `cargo xtask test unit suite python-package` | Deterministic Python package API tests | `lib/python`, `bindings/python` |

## Unit Groups

| Command | Suites |
| --- | --- |
| `cargo xtask test unit group whitebox` | `xtask`, `rust-crates`, `rust-bindings`, `browser-package`, and `demos` |
| `cargo xtask test unit group interface` | `api`, `cli`, `node-package`, and `python-package` |
| `cargo xtask test unit group full` | Every deterministic unit suite |

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
