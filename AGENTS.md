# Sipp Agent Instructions

Welcome! This is the primary context and guidance entry point for AI coding agents working in the Sipp repository.

## 1. Quick Navigation & Context

To avoid token bloat, do not read the entire codebase at once. Instead, refer
to the specialized context files in the [.agents/](.agents/README.md)
directory:
- **General Architecture:** Read [.agents/system/architecture.md](.agents/system/architecture.md) to understand the crate boundaries.
- **Build Instructions:** Read [.agents/build/instructions.md](.agents/build/instructions.md) before executing build commands.
- **Troubleshooting:** Start with [docs/sipp/troubleshooting.md](docs/sipp/troubleshooting.md) for setup and launcher issues.

---

## 2. Configuration & Build Files

Configuration and deployment files must stay explicit and human-managed.
Do not generate, render, or synthesize Docker Compose YAML, TOML, env files,
CI YAML, or other build/deployment configuration from code or `xtask`.

- `.env` / `.env.example` files are for secrets only.
- `*.toml` / `*.toml.example` files are application/runtime configuration.
- Docker `*.yml` / `*.yml.example` files own Docker build, image, port, mount,
  healthcheck, and container orchestration settings.
- Tooling may validate or execute an explicitly selected configuration file,
  but must not obscure configuration behind generated artifacts.

---

## 3. Workspace Build & Run Commands

Always use the **`build-orchestrator`** skill when compiling. The repository uses `xtask` to manage C++ dependencies and environment variables.

- **Build Core (Rust only):** `cargo xtask build core`
- **Build Node Bindings:** `cargo xtask build node` (use `--backend vulkan` for GPU accelerated builds)
- **Build Python Bindings:** `cargo xtask build python` (optionally `--backend vulkan`)
- **Build WebAssembly/WebGPU:** `cargo xtask build wasm`
- **Build All Targets:** `cargo xtask build all`
- **Serve A Demo:** `cargo xtask run demos serve chat`
- **Run llama.cpp Backend Ops:** `cargo xtask run llama backend-ops --backend cpu`

---

## 4. Test & Lint Commands

Always use the **`test-runner`** skill when verifying changes.
- **List Tests:** `cargo xtask test list` (see [docs/testing.md](file:///docs/testing.md) for suite contents)
- **Run All Tests:** `cargo xtask test unit group full`
- **xtask Tests:** `cargo xtask test unit suite xtask`
- **White-box Tests:** `cargo xtask test unit suite rust-crates --package <crate_name>`
- **Interface Tests:** `cargo xtask test unit suite node-package --backend cpu`
- **Verify Coverage/Structure:** `cargo xtask test verify --target whitebox`
- **Rust Tests:** `cargo test` (or `cargo test -p <crate_name>` for narrow Rust-only checks)
- **Rust Linting/Formatting:** `cargo clippy` and `cargo fmt`
- **Browser Package TypeScript Tests:** `cargo xtask test unit suite browser-package`
- **Demo TypeScript Tests:** `cargo xtask test unit suite demos`

---

## 5. Pre-Task Check (Style Checker)

Before completing any task, you **MUST** run the **`style-checker`** skill:
- Check git status (`git status --short`) and diffs (`git diff`).
- Inspect [.agents/skills/style-checker/references/style_guidance.md](.agents/skills/style-checker/references/style_guidance.md).
- Apply minimal local fixes directly to any code violating the guidelines.
- Run the narrowest relevant test/validation command to ensure correctness.
