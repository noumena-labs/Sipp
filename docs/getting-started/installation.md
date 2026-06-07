# Installation

CogentLM documents source-based development first. Public package names are the
user-facing package targets for each language surface.

## Prerequisites

- Rust stable and Cargo.
- Node.js 22 and Bun 1.3.11 for browser and Node package work.
- Python 3.9 or newer for Python package work.
- CMake and Ninja for native builds.
- A GGUF model file for local model-backed examples.

Run setup from the repository root when bootstrapping a checkout:

```bash
./setup.sh
# or on Windows
.\setup.ps1
```

After setup, `clm` is a short repo-local alias for `cargo xtask`.

## Build From Source

Use the xtask orchestrator instead of direct build commands when compiling
CogentLM targets. The orchestrator manages native dependencies, backend
toolchains, and package staging.

```bash
cargo xtask build core
cargo xtask build node --backend cpu
cargo xtask build python --backend cpu
cargo xtask build wasm
```

Use `cargo xtask doctor --target core` when diagnosing local toolchain issues.

## Package Targets

| Surface | Public package target | Source location |
| --- | --- | --- |
| Browser | `cogentlm` | `lib/web` |
| Node.js | `cogentlm-server` | `lib/node` |
| Python | `cogentlm` | `lib/python` |
| Rust | `cogentlm` | `lib/rust` |
| Gateway toolkit | `cogentlm-gateway` | `lib/gateway` |

Use the source checkout and xtask build commands above for local development.
